import pandas as pd
import numpy as np
import logging

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

        if "ticker" not in df.columns:
            df = df.copy()
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
            logger.warning("Split detected — repairing.")
            df.loc[extreme, "close"] = np.nan
            df["close"] = df.groupby("ticker")["close"].ffill()

        return df.reset_index(drop=True)

    ########################################################
    # RETURNS
    ########################################################

    @classmethod
    def add_returns(cls, df):

        df = df.sort_values(["ticker", "date"])

        returns = df.groupby("ticker")["close"].pct_change()
        lo, hi = cls.RETURN_CLAMP

        df["return"] = returns.clip(lo, hi)
        df["return_lag1"] = df.groupby("ticker")["return"].shift(1)
        df["return_lag5"] = df.groupby("ticker")["return"].shift(5)
        df["return_lag10"] = df.groupby("ticker")["return"].shift(10)

        df["momentum_20"] = (
            df.groupby("ticker")["close"]
            .pct_change(20)
            .clip(-1, 1)
        )

        return df

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

        return df

    ########################################################
    # REGIME
    ########################################################

    @classmethod
    def add_regime_feature(cls, df):

        rolling_mean = (
            df.groupby("ticker")["volatility_20"]
            .transform(lambda x: x.rolling(60, min_periods=20).mean())
        )

        rolling_std = (
            df.groupby("ticker")["volatility_20"]
            .transform(lambda x: x.rolling(60, min_periods=20).std(ddof=0))
        )

        zscore = (df["volatility_20"] - rolling_mean) / (rolling_std + 1e-9)

        df["regime_feature"] = np.where(zscore > 0.5, 1.0, 0.0)
        df["regime_feature"] = df["regime_feature"].fillna(0.0)

        return df

    ########################################################
    # TECHNICALS
    ########################################################

    @classmethod
    def add_rsi(cls, df):

        df["rsi"] = np.nan

        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date")
            rsi = TechnicalIndicators.rsi(group[["date", "close"]], window=14)
            df.loc[group.index, "rsi"] = rsi.values.astype("float32")

        df["rsi"] = df["rsi"].fillna(50.0)

        return df

    @classmethod
    def add_macd(cls, df):

        df["macd"] = np.nan
        df["macd_signal"] = np.nan

        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date")
            macd, signal = TechnicalIndicators.macd(group[["date", "close"]])
            df.loc[group.index, "macd"] = macd.astype("float32")
            df.loc[group.index, "macd_signal"] = signal.astype("float32")

        df[["macd", "macd_signal"]] = df[
            ["macd", "macd_signal"]
        ].fillna(0.0)

        return df

    @classmethod
    def add_ema(cls, df):

        df = df.sort_values(["ticker", "date"])

        df["ema_10"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=10, adjust=False).mean()
        )

        df["ema_50"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=50, adjust=False).mean()
        )

        df["ema_ratio"] = (
            df["ema_10"] / df["ema_50"]
        ).replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.5, 1.5)

        return df

    ########################################################
    # CROSS-SECTIONAL (DEPRECATION SAFE)
    ########################################################

    @classmethod
    def add_cross_sectional_features(cls, df):

        cross_cols = [
            "momentum_20",
            "return_lag5",
            "rsi",
            "volatility",
            "ema_ratio"
        ]

        df = df.sort_values(["date", "ticker"]).copy()

        for col in cross_cols:

            cs_mean = df.groupby("date")[col].transform("mean")
            cs_std = df.groupby("date")[col].transform(lambda x: x.std(ddof=0))

            degenerate_mask = (cs_std == 0) | (~np.isfinite(cs_std))

            z = (df[col] - cs_mean) / cs_std.replace(0, np.nan)
            z = z.clip(-5, 5)

            rank = df.groupby("date")[col].rank(pct=True)

            z[degenerate_mask] = 0.0
            rank[degenerate_mask] = 0.5

            df[f"{col}_z"] = z.fillna(0.0)
            df[f"{col}_rank"] = rank.fillna(0.5)

        return df

    ########################################################
    # FINAL SANITIZATION
    ########################################################

    @classmethod
    def _final_sanitize(cls, df):

        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isnull().any():
                if col.endswith("_rank"):
                    df[col] = df[col].fillna(0.5)
                elif col.endswith("_z"):
                    df[col] = df[col].fillna(0.0)
                elif "volatility" in col:
                    df[col] = df[col].fillna(cls.VOL_FLOOR)
                else:
                    df[col] = df[col].fillna(0.0)

        if not np.isfinite(df[numeric_cols].to_numpy()).all():
            raise RuntimeError("Non-finite values remain after sanitization.")

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

        df = cls._validate_price_frame(price_df, ticker)

        df = cls.add_returns(df)
        df = cls.add_volatility(df)
        df = cls.add_regime_feature(df)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_ema(df)
        df = cls.add_cross_sectional_features(df)

        df = cls._final_sanitize(df)

        logger.info("Feature pipeline built | rows=%s", len(df))

        return df.reset_index(drop=True)