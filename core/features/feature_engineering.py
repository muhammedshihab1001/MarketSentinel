import pandas as pd
import numpy as np
import logging

from core.schema.feature_schema import (
    validate_feature_schema,
    MODEL_FEATURES
)

from core.indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngineer:

    MIN_ROWS_REQUIRED = 250
    VOL_FLOOR = 1e-4
    SENTIMENT_STD_FLOOR = 0.02
    RETURN_CLAMP = (-0.5, 0.5)
    MERGE_TOLERANCE = pd.Timedelta("1D")
    MIN_CLASS_RATIO = 0.30

    SENTIMENT_COLUMNS = [
        "date",
        "avg_sentiment",
        "news_count",
        "sentiment_std"
    ]

    @staticmethod
    def _normalize_datetime(df):

        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="raise"
        )

        return df

    @staticmethod
    def _enforce_ticker(df, ticker=None):

        df = df.copy()

        if "ticker" not in df.columns:

            if ticker is None:
                raise RuntimeError(
                    "Ticker column missing from dataframe."
                )

            df["ticker"] = ticker

        df["ticker"] = df["ticker"].astype(str)

        if df["ticker"].isna().any():
            raise RuntimeError("NaN ticker detected.")

        return df

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

    @classmethod
    def _build_neutral_sentiment(cls, price_df):

        neutral = price_df[["date"]].copy()

        neutral["avg_sentiment"] = 0.0
        neutral["news_count"] = 0.0
        neutral["sentiment_std"] = cls.SENTIMENT_STD_FLOOR

        logger.warning(
            "Sentiment unavailable — neutral prior injected."
        )

        return neutral

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

        merged["ticker"] = merged["ticker"].astype(str)

        merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0.0)
        merged["news_count"] = merged["news_count"].fillna(0.0)

        merged["sentiment_std"] = (
            merged["sentiment_std"]
            .fillna(cls.SENTIMENT_STD_FLOOR)
            .clip(lower=cls.SENTIMENT_STD_FLOOR)
        )

        merged["sentiment_lag1"] = merged["avg_sentiment"].shift(1)

        return merged

    @classmethod
    def create_training_dataset(cls, df):

        df = df.sort_values(["ticker", "date"]).copy()

        log_close = np.log(df["close"])
        forward = log_close.shift(-1) - log_close

        safe_vol = df["volatility"].clip(lower=cls.VOL_FLOOR)

        risk_adj = (forward / safe_vol).clip(-5, 5)

        rolling_upper = risk_adj.rolling(252, min_periods=126).quantile(0.65)
        rolling_lower = risk_adj.rolling(252, min_periods=126).quantile(0.35)

        df["target"] = np.where(
            risk_adj >= rolling_upper, 1,
            np.where(risk_adj <= rolling_lower, 0, np.nan)
        )

        required = ["target"] + MODEL_FEATURES

        before = len(df)

        df = df.dropna(subset=required)

        survival = len(df) / before

        logger.info(
            "Target survival ratio → %.2f%%",
            survival * 100
        )

        if survival < 0.40:
            raise RuntimeError(
                f"Feature collapse — survival too low ({survival:.2f})"
            )

        class_ratio = df["target"].mean()

        if not (cls.MIN_CLASS_RATIO < class_ratio < 1 - cls.MIN_CLASS_RATIO):
            raise RuntimeError(
                f"Class imbalance detected ({class_ratio:.2f})"
            )

        df["target"] = df["target"].astype("int8")

        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        return df

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
