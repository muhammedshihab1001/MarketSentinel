import pandas as pd
import numpy as np
import logging

from core.schema.feature_schema import MODEL_FEATURES
from core.indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngineer:

    MIN_ROWS_REQUIRED = 100
    VOL_FLOOR = 1e-4
    RETURN_CLAMP = (-0.5, 0.5)
    SPLIT_THRESHOLD = 3.5

    ########################################################
    # DATETIME NORMALIZATION
    ########################################################

    @staticmethod
    def _normalize_datetime(df):

        df = df.copy()

        if "date" not in df.columns:
            raise RuntimeError("Price dataframe requires 'date' column.")

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])

        if "ticker" not in df.columns:
            df["ticker"] = "unknown"

        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ########################################################
    # PRICE VALIDATION
    ########################################################

    @classmethod
    def _validate_price_frame(cls, df, ticker=None):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe empty.")

        if "ticker" not in df.columns:
            df = df.copy()
            df["ticker"] = ticker or "unknown"

        df = cls._normalize_datetime(df)

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError("Insufficient price history.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        returns = df.groupby("ticker")["close"].pct_change().abs()
        extreme = returns > cls.SPLIT_THRESHOLD

        if extreme.any():
            logger.warning("Split detected — repairing.")
            df.loc[extreme, "close"] = np.nan
            df["close"] = df.groupby("ticker")["close"].ffill()

        return df.reset_index(drop=True)

    ########################################################
    # SENTIMENT MERGE
    ########################################################

    @staticmethod
    def merge_price_sentiment(price_df, sentiment_df):

        price_df = price_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"], utc=True)

        if sentiment_df is None or sentiment_df.empty:
            price_df["avg_sentiment"] = 0.0
            price_df["news_count"] = 0
            price_df["sentiment_std"] = 0.0
            return price_df.sort_values("date").reset_index(drop=True)

        sentiment_df = sentiment_df.copy()
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], utc=True)

        merged = pd.merge(
            price_df,
            sentiment_df,
            on="date",
            how="left"
        )

        for col in ["avg_sentiment", "news_count", "sentiment_std"]:
            if col not in merged.columns:
                merged[col] = 0.0

        merged[["avg_sentiment", "news_count", "sentiment_std"]] = \
            merged[["avg_sentiment", "news_count", "sentiment_std"]].fillna(0)

        return merged.sort_values("date").reset_index(drop=True)

    ########################################################
    # RETURNS
    ########################################################

    @classmethod
    def add_returns(cls, df):

        returns = df.groupby("ticker")["close"].pct_change()
        lo, hi = cls.RETURN_CLAMP

        df["return"] = returns.clip(lo, hi)
        df["return_lag1"] = df.groupby("ticker")["return"].shift(1)
        df["return_lag5"] = df.groupby("ticker")["return"].shift(5)
        df["return_lag10"] = df.groupby("ticker")["return"].shift(10)

        df["momentum_20"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x.pct_change(20))
            .clip(-1, 1)
        )

    ########################################################
    # VOLATILITY
    ########################################################

    @classmethod
    def add_volatility(cls, df):

        grp = df.groupby("ticker")["return"]

        df["volatility_5"] = (
            grp.rolling(5, min_periods=5)
            .std(ddof=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        df["volatility_20"] = (
            grp.rolling(20, min_periods=20)
            .std(ddof=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        df["volatility"] = df["volatility_5"]

        for col in ["volatility", "volatility_5", "volatility_20"]:
            df[col] = df[col].fillna(cls.VOL_FLOOR).clip(lower=cls.VOL_FLOOR)

    ########################################################
    # REGIME
    ########################################################

    @classmethod
    def add_regime_feature(cls, df):

        rolling_vol = (
            df.groupby("ticker")["volatility_20"]
            .transform(lambda x: x.rolling(40, min_periods=20).mean())
        )

        median_vol = rolling_vol.median()

        df["regime_feature"] = np.where(
            rolling_vol > median_vol,
            1.0,
            0.0
        )

    ########################################################
    # TECHNICALS
    ########################################################

    @classmethod
    def add_rsi(cls, df):

        def _compute(group):
            rsi = TechnicalIndicators.rsi(
                group[["date", "close"]],
                window=14
            )
            group["rsi"] = rsi.values.astype("float32")
            return group

        return df.groupby("ticker", group_keys=False).apply(_compute)

    @classmethod
    def add_macd(cls, df):

        def _block(x):
            macd, signal = TechnicalIndicators.macd(
                x[["date", "close"]]
            )
            x["macd"] = macd.astype("float32")
            x["macd_signal"] = signal.astype("float32")
            return x

        return df.groupby("ticker", group_keys=False).apply(_block)

    @classmethod
    def add_ema(cls, df):

        df["ema_10"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x.ewm(span=10, adjust=False).mean())
        )

        df["ema_50"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x.ewm(span=50, adjust=False).mean())
        )

        df["ema_ratio"] = (
            df["ema_10"] / df["ema_50"]
        ).replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.5, 1.5)

    ########################################################
    # CROSS-SECTIONAL FEATURES
    ########################################################

    @staticmethod
    def add_cross_sectional_features(df):

        base_cols = [
            "momentum_20",
            "return_lag5",
            "rsi",
            "volatility",
            "ema_ratio"
        ]

        for col in base_cols:

            mean = df.groupby("date")[col].transform("mean")
            std = df.groupby("date")[col].transform("std").replace(0, 1e-6)

            df[f"{col}_z"] = ((df[col] - mean) / std).astype("float32")

            df[f"{col}_rank"] = (
                df.groupby("date")[col]
                .rank(pct=True)
                .astype("float32")
            )

        return df

    ########################################################
    # MAIN PIPELINE
    ########################################################

    @classmethod
    def build_feature_pipeline(
        cls,
        price_df,
        sentiment_df=None,
        training=False,
        ticker=None
    ):

        if sentiment_df is not None:
            price_df = cls.merge_price_sentiment(price_df, sentiment_df)

        df = cls._validate_price_frame(price_df, ticker)

        cls.add_returns(df)
        cls.add_volatility(df)
        cls.add_regime_feature(df)

        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        cls.add_ema(df)

        df = cls.add_cross_sectional_features(df)

        df = df.dropna()

        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        logger.info("Feature pipeline built | rows=%s", len(df))

        return df.reset_index(drop=True)