"""
MarketSentinel v4.3.0

Feature engineering pipeline for ML model training and inference.
"""

import logging
import numpy as np
import pandas as pd

from core.indicators.technical_indicators import TechnicalIndicators
from core.schema.feature_schema import BASE_CS_COLS, MODEL_FEATURES

logger = logging.getLogger(__name__)


class FeatureEngineer:

    MIN_ROWS_REQUIRED = 120
    SPLIT_THRESHOLD = 3.5

    VOL_FLOOR = 1e-4
    RETURN_CLAMP = (-0.5, 0.5)
    EPSILON = 1e-9
    MIN_CS_WIDTH = 5
    Z_CLIP = 5.0
    MIN_VOLUME = 1.0
    VAR_FLOOR = 1e-6
    DISPERSION_FLOOR = 1e-6

    CS_FEATURE_COLS = list(BASE_CS_COLS)

    # -----------------------------------------------------
    # DATETIME NORMALIZATION
    # -----------------------------------------------------

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["date"] = (
            pd.to_datetime(df["date"], utc=True, errors="coerce")
            .dt.normalize()
        )

        df = df.dropna(subset=["date"])

        df = df.sort_values(["ticker", "date"])

        df = df.drop_duplicates(["ticker", "date"])

        return df.reset_index(drop=True)

    # -----------------------------------------------------
    # PRICE VALIDATION
    # -----------------------------------------------------

    @classmethod
    def _validate_price_frame(cls, df: pd.DataFrame, ticker: str = None):

        if df is None or df.empty:
            raise RuntimeError("Price DataFrame is empty.")

        df = df.copy()

        if "ticker" not in df.columns:
            df["ticker"] = ticker or "unknown"

        df = cls._normalize_datetime(df)

        if "close" not in df.columns:
            raise RuntimeError("Missing required column: 'close'.")

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce")

        df = df.dropna(subset=["close"])

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError(
                f"Insufficient price history: {len(df)} rows."
            )

        if (df["close"] <= 0).any():
            raise RuntimeError("Non-positive close prices detected.")

        returns = df.groupby("ticker")["close"].pct_change().abs()

        extreme = returns > cls.SPLIT_THRESHOLD

        if extreme.any():

            logger.warning(
                "Split-like price move detected in %d bar(s)",
                extreme.sum(),
            )

        df["volume"] = df["volume"].fillna(0).clip(lower=0)

        return df.reset_index(drop=True)

    # -----------------------------------------------------
    # CORE FEATURES
    # -----------------------------------------------------

    @classmethod
    def add_core_features(cls, df):

        df = df.sort_values(["ticker", "date"]).copy()

        raw_returns = df.groupby("ticker")["close"].pct_change()
        raw_returns = raw_returns.replace([np.inf, -np.inf], np.nan).fillna(0)

        returns = raw_returns.groupby(df["ticker"]).transform(
            lambda x: x.rolling(3, min_periods=1).median()
        )

        df["return"] = returns.clip(*cls.RETURN_CLAMP)

        df["return_lag1"] = df.groupby("ticker")["return"].shift(1)
        df["return_lag5"] = df.groupby("ticker")["return"].shift(5)
        df["reversal_5"] = -df["return_lag5"]

        df["return_mean_20"] = (
            df.groupby("ticker")["return"]
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
            .shift(1)
        ).clip(-0.2, 0.2)

        df["momentum_20"] = (
            df.groupby("ticker")["close"].pct_change(20).shift(1)
        )

        df["momentum_60"] = (
            df.groupby("ticker")["close"].pct_change(60).shift(1)
        )

        df["momentum_composite"] = (
            0.6 * df["momentum_20"] + 0.4 * df["momentum_60"]
        ).clip(-2, 2)

        df["volatility_20"] = (
            df.groupby("ticker")["return"]
            .transform(lambda x: x.rolling(20, min_periods=20).std(ddof=0))
            .shift(1)
        )

        df["volatility"] = (
            df["volatility_20"].fillna(cls.VOL_FLOOR).clip(lower=cls.VOL_FLOOR)
        )

        df["mom_vol_adj"] = (
            df["momentum_composite"] / (df["volatility"] + cls.EPSILON)
        ).clip(-5, 5)

        df["vol_of_vol"] = (
            df.groupby("ticker")["volatility"]
            .transform(lambda x: x.rolling(20, min_periods=10).std(ddof=0))
            .shift(1)
        ).fillna(0).clip(0, 1)

        df["return_skew_20"] = (
            df.groupby("ticker")["return"]
            .transform(lambda x: x.rolling(20, min_periods=10).skew())
            .shift(1)
        ).fillna(0).clip(-3, 3)

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

        abs_ret = df["return"].abs()
        daily_illiq = abs_ret / (df["dollar_volume"] + cls.EPSILON)

        df["amihud"] = (
            daily_illiq.groupby(df["ticker"])
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
            .shift(1)
        ).fillna(0)

        df["amihud"] = np.log1p(df["amihud"] * 1e6).clip(0, 20)

        df["rsi"] = 50.0

        for ticker, group in df.groupby("ticker"):

            try:

                rsi_vals = TechnicalIndicators.rsi(
                    group[["date", "close"]], 14
                )

                df.loc[group.index, "rsi"] = rsi_vals.values

            except Exception as exc:

                logger.warning(
                    "RSI failed for %s: %s",
                    ticker,
                    exc,
                )

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

        vol_mean = df.groupby("ticker")["volatility"].transform(
            lambda x: x.rolling(60, min_periods=20).mean()
        )

        vol_std = df.groupby("ticker")["volatility"].transform(
            lambda x: x.rolling(60, min_periods=20).std()
        )

        df["regime_feature"] = (
            (df["volatility"] - vol_mean) / (vol_std + cls.EPSILON)
        ).clip(-3, 3).fillna(0)

        df["momentum_regime_interaction"] = (
            df["momentum_composite"] * df["regime_feature"]
        ).clip(-5, 5)

        return df

    # -----------------------------------------------------
    # CROSS SECTIONAL FEATURES
    # -----------------------------------------------------

    @classmethod
    def add_cross_sectional_features(cls, df):

        df = df.sort_values(["date", "ticker"]).copy()

        counts = df.groupby("date")["ticker"].transform("count")

        df = df[counts >= cls.MIN_CS_WIDTH].copy()

        if df.empty:
            logger.warning("Cross-sectional width insufficient.")
            return df

        df["market_dispersion"] = (
            df.groupby("date")["return"].transform("std")
            .clip(lower=cls.DISPERSION_FLOOR)
            .fillna(0)
        )

        df["breadth"] = (
            df.groupby("date")["return"]
            .transform(lambda x: (x > 0).mean())
            .fillna(0.5)
        )

        cols_to_process = [
            c for c in cls.CS_FEATURE_COLS if c in df.columns
        ]

        for col in cols_to_process:

            lower = df.groupby("date")[col].transform(
                lambda x: x.quantile(0.01)
            )

            upper = df.groupby("date")[col].transform(
                lambda x: x.quantile(0.99)
            )

            clipped = df[col].clip(lower, upper)

            cs_mean = clipped.groupby(df["date"]).transform("mean")

            cs_std = (
                clipped.groupby(df["date"])
                .transform("std")
                .clip(lower=cls.VAR_FLOOR)
            )

            df[f"{col}_z"] = (
                ((clipped - cs_mean) / (cs_std + cls.EPSILON))
                .fillna(0)
                .clip(-cls.Z_CLIP, cls.Z_CLIP)
            )

            df[f"{col}_rank"] = (
                clipped.groupby(df["date"])
                .rank(pct=True)
                .fillna(0.5)
            )

        return df

    # -----------------------------------------------------
    # FINALIZE
    # -----------------------------------------------------

    @classmethod
    def finalize(cls, df):

        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:

            if df[col].isna().any():

                median_fill = df.groupby("ticker")[col].transform(
                    lambda x: x.rolling(20, min_periods=1).median()
                )

                df[col] = df[col].fillna(median_fill)

            df[col] = df[col].fillna(0)

        missing = set(MODEL_FEATURES) - set(df.columns)

        for col in sorted(missing):

            logger.warning("Missing feature %s — filling with 0.", col)

            df[col] = 0.0

        df = df.loc[:, list(set(df.columns) | set(MODEL_FEATURES))]

        return df.reset_index(drop=True)

    # -----------------------------------------------------
    # PIPELINE
    # -----------------------------------------------------

    @classmethod
    def build_feature_pipeline(cls, price_df, training=True):

        df = cls._validate_price_frame(price_df)

        df = cls.add_core_features(df)

        df = cls.add_cross_sectional_features(df)

        df = cls.finalize(df)

        logger.info(
            "Feature pipeline complete | rows=%d tickers=%d features=%d",
            len(df),
            df["ticker"].nunique(),
            len([c for c in df.columns if c in MODEL_FEATURES]),
        )

        return df