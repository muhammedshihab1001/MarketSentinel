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

    MIN_ROWS_REQUIRED = 200
    VOL_FLOOR = 1e-4
    RETURN_CLAMP = (-0.5, 0.5)

    MAX_INDICATOR_WINDOW = 60
    SPLIT_THRESHOLD = 3.5

    FORWARD_DAYS = 5  # 🔥 UPGRADE: 5-day horizon

    ########################################################
    # PRICE VALIDATION
    ########################################################

    @staticmethod
    def _normalize_datetime(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        return df.sort_values("date").reset_index(drop=True)

    @classmethod
    def _validate_price_frame(cls, df, ticker=None):

        if df is None or df.empty:
            raise RuntimeError("Price dataframe empty.")

        df = cls._normalize_datetime(df)

        if "ticker" not in df.columns:
            df["ticker"] = ticker or "unknown"

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

        returns = df.groupby("ticker")["close"].pct_change()

        lo, hi = cls.RETURN_CLAMP
        df["return"] = returns.clip(lo, hi)

        df["return_lag1"] = df.groupby("ticker")["return"].shift(1)
        df["return_lag5"] = df.groupby("ticker")["return"].shift(5)
        df["return_lag10"] = df.groupby("ticker")["return"].shift(10)

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
            df[col] = (
                df[col]
                .fillna(cls.VOL_FLOOR)
                .clip(lower=cls.VOL_FLOOR)
            )

    ########################################################
    # RSI
    ########################################################

    @classmethod
    def add_rsi(cls, df):

        def _compute_rsi(group):
            rsi_series = TechnicalIndicators.rsi(
                group[["date", "close"]],
                window=14
            )

            if isinstance(rsi_series, pd.DataFrame):
                rsi_series = rsi_series.iloc[:, 0]

            group["rsi"] = rsi_series.clip(0, 100).fillna(50).values
            return group

        return df.groupby("ticker", group_keys=False).apply(_compute_rsi)

    ########################################################
    # MACD
    ########################################################

    @classmethod
    def add_macd(cls, df):

        def _macd_block(x):
            macd, signal = TechnicalIndicators.macd(
                x[["date", "close"]]
            )

            x["macd"] = macd.clip(-500, 500).fillna(0)
            x["macd_signal"] = signal.clip(-500, 500).fillna(0)
            return x

        return df.groupby("ticker", group_keys=False).apply(_macd_block)

    ########################################################
    # EMA
    ########################################################

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
    # DATASET BUILDER (5-DAY FORWARD RETURN)
    ########################################################

    @classmethod
    def create_training_dataset(cls, df):

        df = df.sort_values(["ticker", "date"]).copy()

        # 🔥 5-day forward log return
        df["forward_return"] = (
            np.log(df["close"].shift(-cls.FORWARD_DAYS))
            - np.log(df["close"])
        )

        df = df.dropna(subset=["forward_return"])
        df = df.dropna(subset=[*MODEL_FEATURES])

        if len(df) < cls.MIN_ROWS_REQUIRED:
            raise RuntimeError("Feature collapse — dataset too small.")

        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        return df

    ########################################################
    # TRIM WARMUP
    ########################################################

    @classmethod
    def _trim_warmup(cls, df):

        return (
            df.groupby("ticker", group_keys=False)
            .apply(lambda g: g.iloc[cls.MAX_INDICATOR_WINDOW:])
            .reset_index(drop=True)
        )

    ########################################################
    # MAIN PIPELINE
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
            raise RuntimeError("Inference pipeline separate.")

        df = cls._validate_price_frame(price_df, ticker)

        cls.add_returns(df)
        cls.add_volatility(df)

        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        cls.add_ema(df)

        df = cls._trim_warmup(df)
        df = cls.create_training_dataset(df)

        validated = validate_feature_schema(
            df.loc[:, MODEL_FEATURES]
        )

        final = pd.concat(
            [
                df[["date", "close", "forward_return", "ticker"]].reset_index(drop=True),
                validated.reset_index(drop=True)
            ],
            axis=1
        )

        logger.info("Feature pipeline built | rows=%s", len(final))

        return final.reset_index(drop=True)
