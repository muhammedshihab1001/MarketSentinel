import pandas as pd
import numpy as np
import logging

from core.indicators.technical_indicators import TechnicalIndicators
from core.schema.feature_schema import MODEL_FEATURES

logger = logging.getLogger(__name__)


class FeatureEngineer:

    MIN_ROWS_REQUIRED = 100
    VOL_FLOOR = 1e-4
    RETURN_CLAMP = (-0.5, 0.5)
    SPLIT_THRESHOLD = 3.5
    EPSILON = 1e-9

    ########################################################
    # DATETIME NORMALIZATION
    ########################################################

    @staticmethod
    def _normalize_datetime(df):

        df = df.copy()

        if "date" not in df.columns:
            raise RuntimeError("Price dataframe requires 'date' column.")

        df["date"] = (
            pd.to_datetime(df["date"], utc=True, errors="coerce")
            .dt.tz_convert(None)
            .dt.normalize()
        )

        df = df.dropna(subset=["date"])

        if "ticker" not in df.columns:
            df["ticker"] = "unknown"

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        return df

    ########################################################
    # PRICE VALIDATION
    ########################################################

    @classmethod
    def _validate_price_frame(cls, df, ticker=None):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe empty.")

        df = df.copy()

        if "ticker" not in df.columns:
            df["ticker"] = ticker or "unknown"

        df = cls._normalize_datetime(df)

        if "close" not in df.columns:
            raise RuntimeError("Price dataframe requires 'close' column.")

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError("Insufficient price history.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        returns = df.groupby("ticker")["close"].pct_change().abs()
        extreme = returns > cls.SPLIT_THRESHOLD

        if extreme.any():
            logger.warning("Split-like move detected — attempting repair.")
            df.loc[extreme, "close"] = np.nan
            df["close"] = df.groupby("ticker")["close"].ffill()

        df = df.dropna(subset=["close"])

        return df.reset_index(drop=True)

    ########################################################
    # CORE FEATURES
    ########################################################

    @classmethod
    def add_core_features(cls, df):

        df = df.sort_values(["ticker", "date"]).copy()

        returns = df.groupby("ticker")["close"].pct_change()
        lo, hi = cls.RETURN_CLAMP

        df["return"] = returns.clip(lo, hi)

        df["return_lag1"] = df.groupby("ticker")["return"].shift(1)
        df["return_lag5"] = df.groupby("ticker")["return"].shift(5)
        df["return_lag10"] = df.groupby("ticker")["return"].shift(10)

        ########################################################
        # Enhanced Momentum Stack
        ########################################################

        df["momentum_5"] = df.groupby("ticker")["close"].pct_change(5).clip(-1, 1)
        df["momentum_10"] = df.groupby("ticker")["close"].pct_change(10).clip(-1.5, 1.5)
        df["momentum_20"] = df.groupby("ticker")["close"].pct_change(20).clip(-1, 1)
        df["momentum_60"] = df.groupby("ticker")["close"].pct_change(60).clip(-2, 2)

        # Acceleration (new alpha signal)
        df["momentum_accel"] = (
            df["momentum_10"] - df["momentum_20"]
        ).clip(-2, 2)

        grp = df.groupby("ticker")["return"]

        df["volatility_5"] = (
            grp.rolling(5, min_periods=5).std(ddof=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        df["volatility_20"] = (
            grp.rolling(20, min_periods=20).std(ddof=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        df["volatility"] = df["volatility_5"]

        for col in ["volatility", "volatility_5", "volatility_20"]:
            df[col] = (
                df[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(cls.VOL_FLOOR)
                .clip(lower=cls.VOL_FLOOR)
            )

        ########################################################
        # RSI
        ########################################################

        df["rsi"] = np.nan

        for ticker, group in df.groupby("ticker"):
            try:
                rsi = TechnicalIndicators.rsi(group[["date", "close"]], 14)
                df.loc[group.index, "rsi"] = rsi.values.astype("float32")
            except Exception:
                df.loc[group.index, "rsi"] = 50.0

        df["rsi"] = df["rsi"].fillna(50.0).clip(0, 100)

        ########################################################
        # MACD
        ########################################################

        df["macd"] = np.nan
        df["macd_signal"] = np.nan

        for ticker, group in df.groupby("ticker"):
            try:
                macd, signal = TechnicalIndicators.macd(group[["date", "close"]])
                df.loc[group.index, "macd"] = macd.astype("float32")
                df.loc[group.index, "macd_signal"] = signal.astype("float32")
            except Exception:
                df.loc[group.index, "macd"] = 0.0
                df.loc[group.index, "macd_signal"] = 0.0

        df["macd_hist"] = df["macd"] - df["macd_signal"]

        df[["macd", "macd_signal", "macd_hist"]] = (
            df[["macd", "macd_signal", "macd_hist"]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        ########################################################
        # EMA FEATURES
        ########################################################

        df["ema_10"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=10, adjust=False).mean()
        )

        df["ema_50"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=50, adjust=False).mean()
        )

        df["ema_ratio"] = (
            df["ema_10"] / (df["ema_50"] + cls.EPSILON)
        ).replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.5, 1.5)

        ########################################################
        # Volatility Regime (continuous)
        ########################################################

        rolling_mean = (
            df.groupby("ticker")["volatility_20"]
            .transform(lambda x: x.rolling(60, min_periods=20).mean())
        )

        rolling_std = (
            df.groupby("ticker")["volatility_20"]
            .transform(lambda x: x.rolling(60, min_periods=20).std(ddof=0))
        )

        zscore = (df["volatility_20"] - rolling_mean) / (
            rolling_std + cls.EPSILON
        )

        df["regime_feature"] = zscore.clip(-3, 3).fillna(0.0)

        return df

    ########################################################
    # CROSS SECTIONAL
    ########################################################

    @classmethod
    def add_cross_sectional_features(cls, df):

        df = df.sort_values(["date", "ticker"]).copy()

        base_cols = [
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "momentum_60",
            "momentum_accel",
            "return_lag5",
            "rsi",
            "volatility",
            "ema_ratio",
            "macd_hist",
            "regime_feature",
        ]

        single_ticker = df["ticker"].nunique() <= 1

        for col in base_cols:

            if col not in df.columns:
                raise RuntimeError(f"Missing base feature: {col}")

            if single_ticker:
                df[f"{col}_z"] = 0.0
                df[f"{col}_rank"] = 0.5
                continue

            cs_mean = df.groupby("date")[col].transform("mean")
            cs_std = df.groupby("date")[col].transform("std")

            z = (df[col] - cs_mean) / (cs_std.replace(0, np.nan))
            rank = df.groupby("date")[col].rank(method="first", pct=True)

            df[f"{col}_z"] = (
                z.replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(-5, 5)
            )

            df[f"{col}_rank"] = rank.fillna(0.5)

        return df

    ########################################################
    # FINALIZE
    ########################################################

    @classmethod
    def finalize(cls, df):

        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            df[col] = df[col].fillna(0.0)

        if not np.isfinite(df[numeric_cols].to_numpy()).all():
            raise RuntimeError("Non-finite values remain.")

        missing = set(MODEL_FEATURES) - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing features: {missing}")

        return df.reset_index(drop=True)

    ########################################################
    # PIPELINE
    ########################################################

    @classmethod
    def build_feature_pipeline(
        cls,
        price_df,
        sentiment_df=None,
        training=True,
    ):

        df = cls._validate_price_frame(price_df)
        df = cls.add_core_features(df)
        df = cls.add_cross_sectional_features(df)
        df = cls.finalize(df)

        return df