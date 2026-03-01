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
    Z_CLIP = 5.0
    MIN_VOLUME = 1.0

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

        df["volume"] = df["volume"].fillna(0)
        df["volume"] = df["volume"].clip(lower=0)

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

        returns = df.groupby("ticker")["close"].pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)

        df["return"] = returns.clip(*cls.RETURN_CLAMP)

        df["return_lag1"] = df.groupby("ticker")["return"].shift(1)
        df["return_lag5"] = df.groupby("ticker")["return"].shift(5)

        df["reversal_5"] = -df["return_lag5"]

        df["return_mean_20"] = (
            df.groupby("ticker")["return"]
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
            .shift(1)
        ).clip(-0.2, 0.2)

        df["momentum_20"] = df.groupby("ticker")["close"].pct_change(20).shift(1)
        df["momentum_60"] = df.groupby("ticker")["close"].pct_change(60).shift(1)

        df["momentum_composite"] = (
            0.6 * df["momentum_20"] +
            0.4 * df["momentum_60"]
        ).clip(-2, 2)

        grp = df.groupby("ticker")["return"]

        df["volatility_20"] = (
            grp.rolling(20, min_periods=20)
            .std(ddof=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        df["volatility"] = (
            df["volatility_20"]
            .fillna(cls.VOL_FLOOR)
            .clip(lower=cls.VOL_FLOOR)
        )

        df["mom_vol_adj"] = (
            df["momentum_composite"] /
            (df["volatility"] + cls.EPSILON)
        ).clip(-5, 5)

        df["vol_of_vol"] = (
            df.groupby("ticker")["volatility"]
            .transform(lambda x: x.rolling(20, min_periods=5).std())
        ).fillna(0)

        df["return_skew_20"] = (
            df.groupby("ticker")["return"]
            .transform(lambda x: x.rolling(20, min_periods=5).skew())
        ).fillna(0)

        df["volume"] = df["volume"].clip(lower=cls.MIN_VOLUME)

        df["volume_mean_20"] = (
            df.groupby("ticker")["volume"]
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
            .shift(1)
        )

        df["volume_momentum"] = (
            df["volume"] / (df["volume_mean_20"] + cls.EPSILON)
        ).clip(0, 5)

        df["dollar_volume"] = df["close"] * df["volume"]

        df["rsi"] = 50.0
        for ticker, group in df.groupby("ticker"):
            try:
                rsi = TechnicalIndicators.rsi(group[["date", "close"]], 14)
                df.loc[group.index, "rsi"] = rsi.values
            except Exception:
                pass

        df["rsi"] = df["rsi"].clip(0, 100)

        df["ema_10"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=10, adjust=False).mean()
        )
        df["ema_50"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=50, adjust=False).mean()
        )

        df["ema_ratio"] = (
            df["ema_10"] / (df["ema_50"] + cls.EPSILON)
        ).clip(0.5, 1.5)

        rolling_high = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x.rolling(252, min_periods=60).max())
            .shift(1)
        )

        df["dist_from_52w_high"] = (
            (df["close"] / (rolling_high + cls.EPSILON)) - 1
        ).clip(-1, 0)

        df["amihud"] = (
            df["return"].abs() / (df["dollar_volume"] + cls.EPSILON)
        ).clip(0, 1)

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

        df["momentum_regime_interaction"] = (
            df["momentum_composite"] * df["regime_feature"]
        ).clip(-5, 5)

        return df

    ########################################################
    # CROSS SECTIONAL FEATURES
    ########################################################

    @classmethod
    def add_cross_sectional_features(cls, df):

        df = df.sort_values(["date", "ticker"]).copy()

        valid_mask = (
            df.groupby("date")["ticker"]
            .transform("count") >= cls.MIN_CS_WIDTH
        )

        df = df[valid_mask].copy()

        if df.empty:
            raise RuntimeError("Cross-sectional width insufficient.")

        df["market_dispersion"] = (
            df.groupby("date")["return"]
            .transform("std")
            .fillna(0)
        )

        df["breadth"] = (
            df.groupby("date")["return"]
            .transform(lambda x: (x > 0).mean())
            .fillna(0.5)
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:

            lower = df.groupby("date")[col].transform(lambda x: x.quantile(0.01))
            upper = df.groupby("date")[col].transform(lambda x: x.quantile(0.99))
            clipped = df[col].clip(lower, upper)

            cs_mean = clipped.groupby(df["date"]).transform("mean")
            cs_std = clipped.groupby(df["date"]).transform("std")

            z = (clipped - cs_mean) / (cs_std + cls.EPSILON)
            df[f"{col}_z"] = z.fillna(0.0).clip(-cls.Z_CLIP, cls.Z_CLIP)

            ranks = clipped.groupby(df["date"]).rank(pct=True)
            df[f"{col}_rank"] = ranks.fillna(0.5).clip(0.0, 1.0)

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
        return df