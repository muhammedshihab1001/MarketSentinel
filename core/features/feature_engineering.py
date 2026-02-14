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

    MIN_ROWS_REQUIRED = 250
    ABS_FEATURE_LIMIT = 1e5
    VOL_FLOOR = 1e-4
    SENTIMENT_STD_FLOOR = 0.02
    RETURN_CLAMP = (-0.5, 0.5)
    MERGE_TOLERANCE = pd.Timedelta("2D")

    SENTIMENT_COLUMNS = [
        "date",
        "avg_sentiment",
        "news_count",
        "sentiment_std"
    ]

    ###################################################
    # DATETIME
    ###################################################

    @staticmethod
    def _normalize_datetime(df):

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
    def _enforce_ticker(df, ticker=None):

        df = df.copy()

        if "ticker" not in df.columns:

            if ticker is None:
                raise RuntimeError("Ticker column missing.")

            df["ticker"] = ticker

        df["ticker"] = df["ticker"].astype(str)

        return df

    ###################################################
    # PRICE VALIDATION
    ###################################################

    @classmethod
    def _validate_price_frame(cls, df, ticker=None):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe empty.")

        df = cls._enforce_ticker(df, ticker)
        df = cls._normalize_datetime(df)

        required = {"date", "close", "ticker"}

        if not required.issubset(df.columns):
            raise RuntimeError("Price schema violation.")

        df = df.sort_values(["ticker", "date"])

        if df.duplicated(["ticker", "date"]).any():
            raise RuntimeError("Duplicate timestamps.")

        df["close"] = pd.to_numeric(df["close"], errors="raise")

        if not np.isfinite(df["close"]).all():
            raise RuntimeError("Non-finite prices.")

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
            vol.shift(1)
            .clip(lower=FeatureEngineer.VOL_FLOOR)
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
    # GLOBAL SANITIZER (🔥 VERY IMPORTANT)
    ###################################################

    @classmethod
    def _sanitize_features(cls, df):

        feature_block = df.loc[:, MODEL_FEATURES]

        arr = feature_block.to_numpy()

        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite feature values detected.")

        if np.abs(arr).max() > cls.ABS_FEATURE_LIMIT:
            raise RuntimeError("Feature explosion detected.")

        return df

    ###################################################
    # NEUTRAL SENTIMENT
    ###################################################

    @classmethod
    def _build_neutral_sentiment(cls, price_df):

        neutral = price_df[["date"]].copy()

        neutral["avg_sentiment"] = 0.0
        neutral["news_count"] = 0.0
        neutral["sentiment_std"] = cls.SENTIMENT_STD_FLOOR

        logger.warning(
            "Sentiment unavailable — deterministic neutral prior injected."
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
        ticker=None
    ):

        price = cls._normalize_datetime(
            cls._enforce_ticker(price_df, ticker)
        )

        if sentiment_df is None or sentiment_df.empty:
            sentiment_df = cls._build_neutral_sentiment(price)

        sentiment = sentiment_df.loc[:, cls.SENTIMENT_COLUMNS]
        sentiment = cls._normalize_datetime(sentiment)

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

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError("Feature collapse detected.")

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

        if not training:
            raise RuntimeError(
                "Inference pipeline must be separate from training."
            )

        price_df = cls._validate_price_frame(price_df, ticker)

        df = price_df.copy()

        cls.add_returns(df)
        cls.add_volatility(df)
        cls.add_rsi(df)
        cls.add_macd(df)

        df = cls.merge_price_sentiment(
            df,
            sentiment_df,
            ticker
        )

        df = cls.create_training_dataset(df)

        df = cls._sanitize_features(df)

        validated = validate_feature_schema(
            df.loc[:, MODEL_FEATURES]
        )

        allowed = {"date", "close", "target", "ticker"}

        final = pd.concat(
            [
                df[[c for c in df.columns if c in allowed]].reset_index(drop=True),
                validated.reset_index(drop=True)
            ],
            axis=1
        )

        logger.info(
            "Feature pipeline built | rows=%s",
            len(final)
        )

        return final.reset_index(drop=True)
