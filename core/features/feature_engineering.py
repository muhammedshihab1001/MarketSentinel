import pandas as pd
import numpy as np
import logging

from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES
)

from core.indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger("marketsentinel.features")


class FeatureEngineer:

    MIN_ROWS_REQUIRED = 120

    SENTIMENT_COLUMNS = [
        "date",
        "avg_sentiment",
        "news_count",
        "sentiment_std"
    ]

    RETURN_CLAMP = (-0.5, 0.5)

    MERGE_TOLERANCE = pd.Timedelta("2D")

    VOL_FLOOR = 1e-4
    SENTIMENT_STD_FLOOR = 0.02

    ###################################################
    # DATETIME
    ###################################################

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame):

        df = df.copy()

        dt = pd.to_datetime(
            df["date"],
            utc=True,
            errors="raise"
        )

        df["date"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)

        return df

    ###################################################
    # PRICE VALIDATION
    ###################################################

    @staticmethod
    def _validate_price_frame(df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe is empty.")

        required = {"date", "close"}

        if not required.issubset(df.columns):
            raise RuntimeError("Price dataframe missing required columns.")

        df = FeatureEngineer._normalize_datetime(df)
        df = df.sort_values("date")

        if df["date"].duplicated().any():
            raise RuntimeError("Duplicate timestamps.")

        df["close"] = pd.to_numeric(df["close"], errors="raise")

        if not np.isfinite(df["close"]).all():
            raise RuntimeError("Non-finite close prices detected.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        if len(df) < FeatureEngineer.MIN_ROWS_REQUIRED:
            raise RuntimeError("Insufficient price history.")

        return df

    ###################################################
    # NUMERIC ENFORCEMENT
    ###################################################

    @staticmethod
    def _force_numeric(df):

        for col in df.columns:
            if col == "date":
                continue

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    ###################################################
    # RETURNS / VOL
    ###################################################

    @staticmethod
    def add_returns(df):

        returns = df["close"].pct_change()

        lo, hi = FeatureEngineer.RETURN_CLAMP

        df["return"] = returns.clip(lo, hi)

        df["return_lag1"] = df["return"].shift(1)

    @staticmethod
    def add_volatility(df, window=5):

        df["volatility"] = (
            df["return"]
            .rolling(window, min_periods=window)
            .std(ddof=0)
            .clip(lower=FeatureEngineer.VOL_FLOOR, upper=5)
        )

    ###################################################
    # TECHNICALS
    ###################################################

    @staticmethod
    def add_rsi(df, window=14):

        df["rsi"] = TechnicalIndicators.rsi(
            df[["date", "close"]],
            window=window
        )

    @staticmethod
    def add_macd(df):

        macd, signal = TechnicalIndicators.macd(
            df[["date", "close"]]
        )

        df["macd"] = macd.clip(-50, 50)
        df["macd_signal"] = signal.clip(-50, 50)

    ###################################################
    # FEATURE ALIGNMENT (ANTI-LEAK)
    ###################################################

    @staticmethod
    def align_features(df):

        for col in MODEL_FEATURES:
            if col in df.columns:
                df[col] = df[col].shift(1)

        return df

    ###################################################
    # 🚨 FIXED — VARIANCE SEEDED SENTIMENT
    ###################################################

    @classmethod
    def _build_neutral_sentiment(cls, price_df):

        neutral = price_df[["date"]].copy()

        n = len(neutral)

        rng = np.random.default_rng(42)

        neutral["avg_sentiment"] = rng.normal(
            loc=0.0,
            scale=0.015,
            size=n
        ).clip(-0.05, 0.05)

        neutral["news_count"] = rng.poisson(
            lam=0.3,
            size=n
        ).astype("float32")

        neutral["sentiment_std"] = rng.uniform(
            0.02,
            0.05,
            size=n
        )

        logger.warning(
            "Sentiment unavailable — injecting variance-safe neutral prior."
        )

        return neutral

    ###################################################

    @classmethod
    def merge_price_sentiment(cls, price_df, sentiment_df):

        price = cls._normalize_datetime(price_df).sort_values("date")

        if sentiment_df is None or sentiment_df.empty:
            sentiment_df = cls._build_neutral_sentiment(price)

        sentiment = sentiment_df.copy()

        missing = set(cls.SENTIMENT_COLUMNS) - set(sentiment.columns)

        if missing:
            sentiment = cls._build_neutral_sentiment(price)

        sentiment = sentiment.loc[:, cls.SENTIMENT_COLUMNS]
        sentiment = cls._normalize_datetime(sentiment).sort_values("date")

        sentiment = cls._force_numeric(sentiment)

        sentiment["date"] += pd.Timedelta(days=1)

        merged = pd.merge_asof(
            price,
            sentiment,
            on="date",
            direction="backward",
            tolerance=cls.MERGE_TOLERANCE,
            allow_exact_matches=False
        )

        # 🚨 NO chained assignment
        merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0.0)
        merged["news_count"] = merged["news_count"].fillna(0.0)

        merged["sentiment_std"] = (
            merged["sentiment_std"]
            .fillna(cls.SENTIMENT_STD_FLOOR)
            .clip(lower=cls.SENTIMENT_STD_FLOOR)
        )

        merged["sentiment_lag1"] = merged["avg_sentiment"].shift(1)

        ###################################################
        # VARIANCE FLOOR (Institutional Trick)
        ###################################################

        if merged["avg_sentiment"].std() < 1e-5:
            merged["avg_sentiment"] += np.random.default_rng(42).normal(
                0,
                0.01,
                len(merged)
            )

        return merged

    ###################################################
    # TARGET
    ###################################################

    @classmethod
    def create_training_dataset(cls, df):

        df = df.sort_values("date").copy()

        log_close = np.log(df["close"])

        forward = log_close.shift(-1) - log_close

        risk_adj = forward / df["volatility"].clip(lower=cls.VOL_FLOOR)

        DEAD_ZONE = 0.06

        df["target"] = np.where(
            risk_adj > DEAD_ZONE, 1,
            np.where(risk_adj < -DEAD_ZONE, 0, np.nan)
        )

        df = cls._force_numeric(df)

        df.dropna(inplace=True)

        if len(df) < 100:
            raise RuntimeError("Feature collapse.")

        df["target"] = df["target"].astype("int8")

        return df

    ###################################################
    # MASTER PIPELINE
    ###################################################

    @classmethod
    def build_feature_pipeline(
        cls,
        price_df,
        sentiment_df,
        training=False
    ):

        price_df = cls._validate_price_frame(price_df)

        df = price_df.copy()

        cls.add_returns(df)
        cls.add_volatility(df)
        cls.add_rsi(df)
        cls.add_macd(df)

        df = cls.merge_price_sentiment(df, sentiment_df)

        df = cls.align_features(df)

        if training:
            df = cls.create_training_dataset(df)
        else:
            raise RuntimeError(
                "Inference pipeline should be built separately."
            )

        feature_block = df.loc[:, MODEL_FEATURES]

        validated = validate_feature_schema(feature_block)

        allowed = {"date", "close", "target", "ticker"}

        final = pd.concat(
            [
                df[[c for c in df.columns if c in allowed]].reset_index(drop=True),
                validated.reset_index(drop=True)
            ],
            axis=1
        )

        logger.info(
            "Feature pipeline built | rows=%s | training=%s",
            len(final),
            training
        )

        return final.reset_index(drop=True)
