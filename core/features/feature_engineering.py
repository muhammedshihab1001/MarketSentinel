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

    TRAINING_SEED = 42

    ###################################################
    # DATETIME
    ###################################################

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame):

        df = df.copy()

        df["date"] = (
            pd.to_datetime(
                df["date"],
                utc=True,
                errors="raise"
            ).dt.tz_convert(None)
        )

        return df

    ###################################################
    # TICKER ENFORCEMENT
    ###################################################

    @staticmethod
    def _enforce_ticker(df: pd.DataFrame, ticker=None):

        df = df.copy()

        if "ticker" not in df.columns:

            if ticker is None:
                raise RuntimeError(
                    "Ticker column missing from price dataframe."
                )

            df["ticker"] = ticker

        df["ticker"] = df["ticker"].astype(str)

        return df

    ###################################################
    # PRICE VALIDATION
    ###################################################

    @classmethod
    def _validate_price_frame(cls, df: pd.DataFrame, ticker=None):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe is empty.")

        df = cls._enforce_ticker(df, ticker)
        df = cls._normalize_datetime(df)

        required = {"date", "close", "ticker"}

        if not required.issubset(df.columns):
            raise RuntimeError("Price dataframe missing required columns.")

        df = df.sort_values(["ticker", "date"])

        if df.duplicated(["ticker", "date"]).any():
            raise RuntimeError("Duplicate ticker-date detected.")

        df["close"] = pd.to_numeric(df["close"], errors="raise")

        if not np.isfinite(df["close"]).all():
            raise RuntimeError("Non-finite close prices detected.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError("Insufficient price history.")

        return df.reset_index(drop=True)

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

        vol = (
            df["return"]
            .rolling(window, min_periods=window)
            .std(ddof=0)
        )

        df["volatility"] = (
            vol.clip(lower=FeatureEngineer.VOL_FLOOR)
            .fillna(FeatureEngineer.VOL_FLOOR)
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
    # FEATURE ALIGNMENT
    ###################################################

    @staticmethod
    def align_features(df):

        df = df.copy()

        for col in MODEL_FEATURES:
            if col in df.columns:
                df[col] = df[col].shift(1)

        # HARD LEAKAGE GUARD
        if "close" in df.columns:
            for col in MODEL_FEATURES:
                if col in df.columns and df[col].equals(df["close"]):
                    raise RuntimeError(
                        f"Leakage detected: feature '{col}' mirrors close."
                    )

        return df

    ###################################################
    # NEUTRAL SENTIMENT
    ###################################################

    @classmethod
    def _build_neutral_sentiment(cls, price_df, training):

        neutral = price_df[["date"]].copy()

        rng = np.random.default_rng(
            cls.TRAINING_SEED if training else None
        )

        n = len(neutral)

        neutral["avg_sentiment"] = (
            rng.normal(0, 0.04, n)
            + rng.normal(0, 0.015, n)
        ).clip(-0.15, 0.15)

        neutral["news_count"] = (
            rng.poisson(3, n)
            + rng.integers(0, 3, n)
        ).astype("float32")

        neutral["sentiment_std"] = rng.uniform(
            0.03, 0.10, n
        )

        logger.warning(
            "Sentiment unavailable — injected stochastic neutral prior."
        )

        return neutral

    ###################################################
    # MERGE
    ###################################################

    @classmethod
    def merge_price_sentiment(
        cls,
        price_df,
        sentiment_df,
        ticker=None,
        training=False
    ):

        price_df = cls._enforce_ticker(price_df, ticker)
        price = cls._normalize_datetime(price_df)

        if sentiment_df is None or sentiment_df.empty:
            sentiment_df = cls._build_neutral_sentiment(price, training)

        sentiment = sentiment_df.loc[:, cls.SENTIMENT_COLUMNS]
        sentiment = cls._normalize_datetime(sentiment)

        # HARD LOOKAHEAD GUARD
        sentiment["date"] += pd.Timedelta(days=1)

        merged = pd.merge_asof(
            price.sort_values("date"),
            sentiment.sort_values("date"),
            on="date",
            direction="backward",
            tolerance=cls.MERGE_TOLERANCE,
            allow_exact_matches=False
        )

        merged["avg_sentiment"].fillna(0.0, inplace=True)
        merged["news_count"].fillna(0.0, inplace=True)

        merged["sentiment_std"] = (
            merged["sentiment_std"]
            .fillna(cls.SENTIMENT_STD_FLOOR)
            .clip(lower=cls.SENTIMENT_STD_FLOOR)
        )

        merged["sentiment_lag1"] = merged["avg_sentiment"].shift(1)

        return merged

    ###################################################
    # TARGET
    ###################################################

    @classmethod
    def create_training_dataset(cls, df):

        df = df.sort_values(["ticker", "date"]).copy()

        log_close = np.log(df["close"])
        forward = log_close.shift(-1) - log_close

        safe_vol = df["volatility"].clip(lower=cls.VOL_FLOOR)

        risk_adj = (forward / safe_vol).clip(-5, 5)

        DEAD_ZONE = 0.06

        df["target"] = np.where(
            risk_adj > DEAD_ZONE, 1,
            np.where(risk_adj < -DEAD_ZONE, 0, np.nan)
        )

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
        training=False,
        ticker=None
    ):

        price_df = cls._validate_price_frame(price_df, ticker)

        df = price_df.copy()

        cls.add_returns(df)
        cls.add_volatility(df)
        cls.add_rsi(df)
        cls.add_macd(df)

        df = cls.merge_price_sentiment(
            df,
            sentiment_df,
            ticker,
            training
        )

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
