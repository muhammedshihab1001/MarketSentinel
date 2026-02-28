import pandas as pd
import numpy as np
import logging

from core.indicators.technical_indicators import TechnicalIndicators
from core.schema.feature_schema import MODEL_FEATURES

logger = logging.getLogger(__name__)


class FeatureEngineer:

    MIN_ROWS_REQUIRED = 120
    VOL_FLOOR = 1e-4
    RETURN_CLAMP = (-0.5, 0.5)
    SPLIT_THRESHOLD = 3.5
    EPSILON = 1e-9
    MIN_CS_WIDTH = 5

    ########################################################
    # DATETIME NORMALIZATION
    ########################################################

    @staticmethod
    def _normalize_datetime(df):
        df = df.copy()

        df["date"] = (
            pd.to_datetime(df["date"], utc=True, errors="coerce")
            .dt.tz_convert(None)
            .dt.normalize()
        )

        df = df.dropna(subset=["date"])
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
            raise RuntimeError("Missing close column.")

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce")

        df = df.dropna(subset=["close"])

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError("Insufficient price history.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        returns = df.groupby("ticker")["close"].pct_change().abs()
        extreme = returns > cls.SPLIT_THRESHOLD

        if extreme.any():
            logger.warning("Split-like move detected — repairing.")
            df.loc[extreme, "close"] = np.nan
            df["close"] = df.groupby("ticker")["close"].ffill()

        return df.dropna(subset=["close"]).reset_index(drop=True)

    ########################################################
    # CORE FEATURES
    ########################################################

    @classmethod
    def add_core_features(cls, df):

        df = df.sort_values(["ticker", "date"]).copy()

        # RETURNS
        returns = df.groupby("ticker")["close"].pct_change()
        df["return"] = returns.clip(*cls.RETURN_CLAMP)

        df["return_lag1"] = df.groupby("ticker")["return"].shift(1)
        df["return_lag5"] = df.groupby("ticker")["return"].shift(5)

        df["return_mean_20"] = (
            df.groupby("ticker")["return"]
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
            .shift(1)
        ).clip(-0.2, 0.2)

        # MOMENTUM
        df["momentum_20"] = df.groupby("ticker")["close"].pct_change(20)
        df["momentum_60"] = df.groupby("ticker")["close"].pct_change(60)

        df["momentum_composite"] = (
            0.6 * df["momentum_20"] +
            0.4 * df["momentum_60"]
        ).clip(-2, 2)

        # VOLUME FEATURES (RESTORED)
        df["volume_mean_20"] = (
            df.groupby("ticker")["volume"]
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
        )

        df["volume_momentum"] = (
            df["volume"] / (df["volume_mean_20"] + cls.EPSILON)
        ).clip(0, 5)

        df["dollar_volume"] = df["close"] * df["volume"]

        # VOLATILITY
        grp = df.groupby("ticker")["return"]

        df["volatility_20"] = (
            grp.rolling(20, min_periods=20).std(ddof=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        df["volatility"] = (
            df["volatility_20"]
            .fillna(cls.VOL_FLOOR)
            .clip(lower=cls.VOL_FLOOR)
        )

        # RSI
        df["rsi"] = 50.0
        for ticker, group in df.groupby("ticker"):
            try:
                rsi = TechnicalIndicators.rsi(group[["date", "close"]], 14)
                df.loc[group.index, "rsi"] = rsi.values
            except Exception:
                pass

        df["rsi"] = df["rsi"].clip(0, 100)

        # EMA
        df["ema_10"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=10, adjust=False).mean()
        )

        df["ema_50"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=50, adjust=False).mean()
        )

        df["ema_ratio"] = (
            df["ema_10"] / (df["ema_50"] + cls.EPSILON)
        ).clip(0.5, 1.5)

        # DISTANCE FROM 52W HIGH
        rolling_high = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x.rolling(252, min_periods=60).max())
            .shift(1)
        )

        df["dist_from_52w_high"] = (
            (df["close"] / (rolling_high + cls.EPSILON)) - 1
        ).clip(-1, 0)

        # AMIHUD
        df["amihud"] = (
            df["return"].abs() / (df["dollar_volume"] + cls.EPSILON)
        ).clip(0, 1)

        # REGIME FEATURE
        vol_mean = (
            df.groupby("ticker")["volatility"]
            .transform(lambda x: x.rolling(60, min_periods=20).mean())
        )

        vol_std = (
            df.groupby("ticker")["volatility"]
            .transform(lambda x: x.rolling(60, min_periods=20).std())
        )

        df["regime_feature"] = (
            (df["volatility"] - vol_mean) /
            (vol_std + cls.EPSILON)
        ).clip(-3, 3).fillna(0.0)

        return df

    ########################################################
    # CROSS SECTIONAL FEATURES
    ########################################################

    @classmethod
    def add_cross_sectional_features(cls, df):

        df = df.sort_values(["date", "ticker"]).copy()

        valid_mask = df.groupby("date")["ticker"].transform("count") >= cls.MIN_CS_WIDTH
        df = df[valid_mask].copy()

        base_cols = [
            "momentum_20",
            "momentum_60",
            "momentum_composite",
            "return_lag5",
            "return_mean_20",
            "rsi",
            "volatility",
            "ema_ratio",
            "volume_momentum",
            "dollar_volume",
            "dist_from_52w_high",
            "regime_feature",
            "amihud",
        ]

        for col in base_cols:

            if col not in df.columns:
                continue

            cs_mean = df.groupby("date")[col].transform("mean")
            cs_std = df.groupby("date")[col].transform("std")

            z = (df[col] - cs_mean) / (cs_std + cls.EPSILON)

            df[f"{col}_z"] = z.fillna(0.0).clip(-5, 5)
            df[f"{col}_rank"] = (
                df.groupby("date")[col]
                .rank(method="first", pct=True)
                .fillna(0.5)
            )

        # MARKET DISPERSION
        df["market_dispersion"] = (
            df.groupby("date")["return"].transform("std").clip(0, 0.2)
        )

        # BREADTH
        df["breadth"] = (
            df.groupby("date")["return"]
            .transform(lambda x: (x > 0).mean())
        )

        for col in ["market_dispersion", "breadth"]:

            cs_mean = df.groupby("date")[col].transform("mean")
            cs_std = df.groupby("date")[col].transform("std")

            z = (df[col] - cs_mean) / (cs_std + cls.EPSILON)

            df[f"{col}_z"] = z.fillna(0.0).clip(-5, 5)
            df[f"{col}_rank"] = (
                df.groupby("date")[col]
                .rank(method="first", pct=True)
                .fillna(0.5)
            )

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

        missing = set(MODEL_FEATURES) - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing features: {missing}")

        if not np.isfinite(df[numeric_cols].to_numpy()).all():
            raise RuntimeError("Non-finite values remain.")

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