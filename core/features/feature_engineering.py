"""
MarketSentinel v4.1.0

Feature engineering pipeline for ML model training and inference.

Pipeline stages:
    1. _validate_price_frame  — schema + basic sanity (no data repair)
    2. add_core_features      — returns, momentum, volatility, RSI, EMA, regime
    3. add_cross_sectional_features — z-scores + ranks for model features only
    4. finalize               — inf removal, median fill, MODEL_FEATURES alignment

Output feeds:
    - XGBoost model (training + inference)
    - SignalAgent   (rsi, ema_ratio, momentum_20_z, regime_feature, etc.)
    - TechnicalRiskAgent (momentum_20_z, ema_ratio, rsi, regime_feature)
"""

import logging

import numpy as np
import pandas as pd

from core.indicators.technical_indicators import TechnicalIndicators
from core.schema.feature_schema import BASE_CS_COLS, MODEL_FEATURES

logger = logging.getLogger(__name__)


class FeatureEngineer:

    # ── Validation thresholds ─────────────────────────────────────────────────
    MIN_ROWS_REQUIRED = 120
    SPLIT_THRESHOLD   = 3.5    # pct_change above this = suspected split

    # ── Feature computation constants ────────────────────────────────────────
    VOL_FLOOR    = 1e-4
    RETURN_CLAMP = (-0.5, 0.5)
    EPSILON      = 1e-9
    MIN_CS_WIDTH = 5         # minimum tickers per date for cross-sectional features
    Z_CLIP       = 5.0
    MIN_VOLUME   = 1.0
    VAR_FLOOR    = 1e-6

    # ── Columns to include in cross-sectional z-scoring ───────────────────────
    # Must match BASE_CS_COLS from feature_schema exactly so that all
    # expected _z and _rank columns are produced.
    CS_FEATURE_COLS = list(BASE_CS_COLS)

    # ────────────────────────────────────────────────────────────────────────
    # DATETIME NORMALISATION  (UTC-aware — no timezone stripping)
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date column to UTC-aware datetime and normalise to midnight.
        Preserves UTC timezone — does NOT strip it.
        """
        df = df.copy()
        df["date"] = (
            pd.to_datetime(df["date"], utc=True, errors="coerce")
            .dt.normalize()
        )
        df = df.dropna(subset=["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    # ────────────────────────────────────────────────────────────────────────
    # PRICE VALIDATION  (schema + sanity — no data repair)
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def _validate_price_frame(
        cls,
        df:     pd.DataFrame,
        ticker: str = None,
    ) -> pd.DataFrame:
        """
        Validate price frame schema and basic sanity.
        Does NOT repair OHLC or extreme moves — that is handled upstream
        in data_fetcher → yahoo_provider → MarketProviderRouter.
        """
        if df is None or df.empty:
            raise RuntimeError("Price DataFrame is empty.")

        df = df.copy()

        if "ticker" not in df.columns:
            df["ticker"] = ticker or "unknown"

        df = cls._normalize_datetime(df)

        if "close" not in df.columns:
            raise RuntimeError("Missing required column: 'close'.")

        df["close"]  = pd.to_numeric(df["close"],                errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0),        errors="coerce")

        df = df.dropna(subset=["close"])

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError(
                f"Insufficient price history: {len(df)} rows, "
                f"need {cls.MIN_ROWS_REQUIRED}."
            )

        if (df["close"] <= 0).any():
            raise RuntimeError("Non-positive close prices detected.")

        # Warn on split-like moves but do NOT repair — flag for investigation
        returns  = df.groupby("ticker")["close"].pct_change().abs()
        extreme  = returns > cls.SPLIT_THRESHOLD
        if extreme.any():
            n_extreme = extreme.sum()
            logger.warning(
                "Split-like price move detected in %d bar(s) — "
                "data repair should be handled upstream.",
                n_extreme,
            )

        df["volume"] = df["volume"].fillna(0).clip(lower=0)

        return df.dropna(subset=["close"]).reset_index(drop=True)

    # ────────────────────────────────────────────────────────────────────────
    # CORE FEATURES
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def add_core_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-ticker features: returns, momentum, volatility,
        RSI, EMA structure, regime.
        All forward-leaking features are lagged by 1 period.
        """
        df = df.sort_values(["ticker", "date"]).copy()

        # ── Returns ───────────────────────────────────────────────────────────
        raw_returns = df.groupby("ticker")["close"].pct_change()
        raw_returns = raw_returns.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Smooth with rolling 3-period median (backward-looking — no lookahead)
        returns = raw_returns.groupby(df["ticker"]).transform(
            lambda x: x.rolling(3, min_periods=1).median()
        )
        df["return"] = returns.clip(*cls.RETURN_CLAMP)

        df["return_lag1"]  = df.groupby("ticker")["return"].shift(1)
        df["return_lag5"]  = df.groupby("ticker")["return"].shift(5)
        df["reversal_5"]   = -df["return_lag5"]

        df["return_mean_20"] = (
            df.groupby("ticker")["return"]
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
            .shift(1)
        ).clip(-0.2, 0.2)

        # ── Momentum ──────────────────────────────────────────────────────────
        df["momentum_20"] = (
            df.groupby("ticker")["close"].pct_change(20).shift(1)
        )
        df["momentum_60"] = (
            df.groupby("ticker")["close"].pct_change(60).shift(1)
        )
        df["momentum_composite"] = (
            0.6 * df["momentum_20"] + 0.4 * df["momentum_60"]
        ).clip(-2, 2)

        # ── Volatility ────────────────────────────────────────────────────────
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

        # ── Volatility of Volatility ──────────────────────────────────────────
        df["vol_of_vol"] = (
            df.groupby("ticker")["volatility"]
            .transform(lambda x: x.rolling(20, min_periods=10).std(ddof=0))
            .shift(1)
        ).fillna(0).clip(0, 1)

        # ── Return Skewness ───────────────────────────────────────────────────
        df["return_skew_20"] = (
            df.groupby("ticker")["return"]
            .transform(lambda x: x.rolling(20, min_periods=10).skew())
            .shift(1)
        ).fillna(0).clip(-3, 3)

        # ── Liquidity ─────────────────────────────────────────────────────────
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

        # ── Amihud Illiquidity ────────────────────────────────────────────────
        # Amihud = mean(|return| / dollar_volume) over rolling window
        # Higher values = less liquid
        abs_ret = df["return"].abs()
        daily_illiq = abs_ret / (df["dollar_volume"] + cls.EPSILON)
        df["amihud"] = (
            daily_illiq.groupby(df["ticker"])
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
            .shift(1)
        ).fillna(0)
        # Log-scale to compress extreme values (penny stocks)
        df["amihud"] = np.log1p(df["amihud"] * 1e6).clip(0, 20)

        # ── RSI ───────────────────────────────────────────────────────────────
        df["rsi"] = 50.0   # safe default

        rsi_failures = 0
        for ticker, group in df.groupby("ticker"):
            try:
                rsi_vals = TechnicalIndicators.rsi(
                    group[["date", "close"]], 14
                )
                df.loc[group.index, "rsi"] = rsi_vals.values
            except Exception as exc:
                rsi_failures += 1
                logger.warning(
                    "RSI calculation failed for ticker=%s: %s — using default 50.",
                    ticker, exc,
                )

        if rsi_failures > 0:
            logger.warning(
                "RSI failed for %d ticker(s) — those tickers use default rsi=50.",
                rsi_failures,
            )

        df["rsi"] = df["rsi"].clip(0, 100)

        # ── EMA structure ─────────────────────────────────────────────────────
        df["ema_10"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=10, adjust=False).mean()
        )
        df["ema_50"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=50, adjust=False).mean()
        )
        df["ema_ratio"] = (
            df["ema_10"] / (df["ema_50"] + cls.EPSILON)
        ).clip(0.5, 1.5)

        # ── 52-week high distance ─────────────────────────────────────────────
        rolling_high = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x.rolling(252, min_periods=60).max())
            .shift(1)
        )
        df["dist_from_52w_high"] = (
            (df["close"] / (rolling_high + cls.EPSILON)) - 1
        ).clip(-1, 0)

        # ── Volatility regime ─────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────────────────
    # CROSS-SECTIONAL FEATURES  (z-scores + ranks on whitelisted cols only)
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def add_cross_sectional_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-sectional z-scores and percentile ranks.

        Only processes CS_FEATURE_COLS (derived from BASE_CS_COLS in
        feature_schema) to ensure all expected _z and _rank columns
        are produced.
        """
        df = df.sort_values(["date", "ticker"]).copy()

        # Drop dates with fewer than MIN_CS_WIDTH tickers
        counts = df.groupby("date")["ticker"].transform("count")
        df     = df[counts >= cls.MIN_CS_WIDTH].copy()

        if df.empty:
            logger.warning(
                "Cross-sectional width insufficient after filtering — "
                "skipping CS features."
            )
            return df

        # ── Market-level context ──────────────────────────────────────────────
        df["market_dispersion"] = (
            df.groupby("date")["return"].transform("std").fillna(0)
        )
        df["breadth"] = (
            df.groupby("date")["return"]
            .transform(lambda x: (x > 0).mean())
            .fillna(0.5)
        )

        # ── Z-scores + ranks for whitelisted feature columns only ─────────────
        cols_to_process = [
            c for c in cls.CS_FEATURE_COLS if c in df.columns
        ]

        missing_cs = [c for c in cls.CS_FEATURE_COLS if c not in df.columns]
        if missing_cs:
            logger.warning(
                "CS columns missing from DataFrame (will not get _z/_rank): %s",
                missing_cs,
            )

        for col in cols_to_process:
            # Winsorise at 1st/99th percentile per date
            lower   = df.groupby("date")[col].transform(lambda x: x.quantile(0.01))
            upper   = df.groupby("date")[col].transform(lambda x: x.quantile(0.99))
            clipped = df[col].clip(lower, upper)

            cs_mean = clipped.groupby(df["date"]).transform("mean")
            cs_std  = clipped.groupby(df["date"]).transform("std").clip(lower=cls.VAR_FLOOR)

            df[f"{col}_z"] = (
                ((clipped - cs_mean) / (cs_std + cls.EPSILON))
                .fillna(0.0)
                .clip(-cls.Z_CLIP, cls.Z_CLIP)
            )

            df[f"{col}_rank"] = (
                clipped.groupby(df["date"])
                .rank(pct=True)
                .fillna(0.5)
            )

        return df

    # ────────────────────────────────────────────────────────────────────────
    # FINALIZE
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def finalize(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final pass:
            - Replace inf with NaN
            - Median-fill NaNs per ticker (rolling window)
            - Zero-fill any remaining NaNs
            - Ensure all MODEL_FEATURES columns exist
        """
        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isna().any():
                median_fill = df.groupby("ticker")[col].transform(
                    lambda x: x.rolling(20, min_periods=1).median()
                )
                df[col] = df[col].fillna(median_fill)
            df[col] = df[col].fillna(0.0)

        # Ensure all expected model features exist
        missing_features = set(MODEL_FEATURES) - set(df.columns)
        for col in sorted(missing_features):
            logger.warning(
                "MODEL_FEATURES column '%s' missing — filling with 0.0.", col
            )
            df[col] = 0.0

        return df.reset_index(drop=True)

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC PIPELINE
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def build_feature_pipeline(
        cls,
        price_df:  pd.DataFrame,
        training:  bool = True,
    ) -> pd.DataFrame:
        """
        Full feature engineering pipeline.

        Parameters
        ----------
        price_df : OHLCV DataFrame from MarketDataService
        training : passed through for future training-specific logic

        Returns
        -------
        Feature DataFrame ready for XGBoost model input.
        """
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