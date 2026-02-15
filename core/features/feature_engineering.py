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

    MIN_ROWS_REQUIRED = 180
    VOL_FLOOR = 1e-4
    SENTIMENT_STD_FLOOR = 0.02

    RETURN_CLAMP = (-0.5, 0.5)

    SPLIT_THRESHOLD = 3.5
    MERGE_TOLERANCE = pd.Timedelta("3D")

    MIN_CLASS_RATIO = 0.15

    SENTIMENT_COLUMNS = [
        "date",
        "avg_sentiment",
        "news_count",
        "sentiment_std"
    ]

    ########################################################
    # TIME
    ########################################################

    @staticmethod
    def _normalize_datetime(df):

        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce"
        )

        df = df.dropna(subset=["date"])

        return df.sort_values("date").reset_index(drop=True)

    ########################################################

    @staticmethod
    def _enforce_ticker(df, ticker=None):

        df = df.copy()

        if "ticker" not in df.columns:
            df["ticker"] = ticker or "unknown"

        df["ticker"] = df["ticker"].astype(str)

        return df

    ########################################################

    @classmethod
    def _validate_price_frame(cls, df, ticker=None):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe empty.")

        df = cls._enforce_ticker(df, ticker)
        df = cls._normalize_datetime(df)

        required = {"date", "close", "ticker"}
        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(f"Price schema violation: {missing}")

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError("Insufficient price history.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        ####################################################
        # SPLIT REPAIR
        ####################################################

        returns = df["close"].pct_change().abs()

        extreme = returns > cls.SPLIT_THRESHOLD

        if extreme.any():
            logger.warning("Split detected — repairing price path.")
            df.loc[extreme, "close"] = np.nan
            df["close"] = df["close"].ffill()

        return df.reset_index(drop=True)

    ########################################################
    # TECHNICALS
    ########################################################

    @staticmethod
    def add_returns(df):

        returns = df["close"].pct_change()

        lo, hi = FeatureEngineer.RETURN_CLAMP

        df["return"] = returns.clip(lo, hi)
        df["return_lag1"] = df["return"].shift(1)

    ########################################################

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

    ########################################################

    @staticmethod
    def add_rsi(df, window=14):

        df["rsi"] = TechnicalIndicators.rsi(
            df[["date", "close"]],
            window=window
        ).clip(0, 100)

    ########################################################

    @staticmethod
    def add_macd(df):

        macd, signal = TechnicalIndicators.macd(
            df[["date", "close"]]
        )

        df["macd"] = macd.clip(-25, 25)
        df["macd_signal"] = signal.clip(-25, 25)

    ########################################################
    # ⭐ INSTITUTIONAL FIX — NO CONSTANT SENTIMENT
    ########################################################

    @classmethod
    def _build_uncertain_sentiment(cls, price_df):

        rows = len(price_df)

        rng = np.random.default_rng(42)  
        # fixed seed = reproducible training

        neutral = price_df[["date"]].copy()

        # small mean-zero noise
        neutral["avg_sentiment"] = rng.normal(
            0.0,
            0.02,
            rows
        ).astype("float32")

        # simulate sparse news arrival
        neutral["news_count"] = rng.poisson(
            0.3,
            rows
        ).astype("float32")

        # ⭐ FIX — NO CONSTANT STD
        neutral["sentiment_std"] = np.abs(
            rng.normal(
                cls.SENTIMENT_STD_FLOOR,
                0.01,
                rows
            )
        ).astype("float32")

        logger.warning(
            "Sentiment unavailable — stochastic distribution injected."
        )

        return neutral


    ########################################################

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
            sentiment_df = cls._build_uncertain_sentiment(price)

        sentiment = sentiment_df.loc[:, cls.SENTIMENT_COLUMNS]
        sentiment = cls._normalize_datetime(sentiment)

        sentiment["date"] += pd.Timedelta(days=1)

        merged = pd.merge_asof(
            price,
            sentiment,
            on="date",
            direction="backward",
            tolerance=cls.MERGE_TOLERANCE
        )

        merged.fillna({
            "avg_sentiment": 0.0,
            "news_count": 0.0,
            "sentiment_std": cls.SENTIMENT_STD_FLOOR
        }, inplace=True)

        merged["sentiment_lag1"] = (
            merged["avg_sentiment"]
            .shift(1)
            .fillna(0)
        )

        return merged

    ########################################################
    # TARGET
    ########################################################

    @classmethod
    def create_training_dataset(cls, df):

        df = df.sort_values(["ticker", "date"]).copy()

        log_close = np.log(df["close"])

        forward = log_close.shift(-1) - log_close

        df = df.iloc[:-1].copy()
        forward = forward.iloc[:-1]

        safe_vol = df["volatility"].clip(lower=cls.VOL_FLOOR)

        risk_adj = (forward / safe_vol).clip(-5, 5)

        upper = risk_adj.quantile(0.66)
        lower = risk_adj.quantile(0.34)

        df["target"] = np.where(
            risk_adj >= upper, 1,
            np.where(risk_adj <= lower, 0, np.nan)
        )

        required = ["target", *MODEL_FEATURES]

        df = df.dropna(subset=required)

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError("Feature collapse — dataset too small.")

        class_ratio = df["target"].mean()

        if not (cls.MIN_CLASS_RATIO < class_ratio < 1 - cls.MIN_CLASS_RATIO):
            logger.warning(
                "Class imbalance detected — continuing (%.2f)",
                class_ratio
            )

        df["target"] = df["target"].astype("int8")

        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        return df

    ########################################################

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

        final = pd.concat(
            [
                df[["date", "close", "target", "ticker"]].reset_index(drop=True),
                validated.reset_index(drop=True)
            ],
            axis=1
        )

        logger.info("Feature pipeline built | rows=%s", len(final))

        return final.reset_index(drop=True)
