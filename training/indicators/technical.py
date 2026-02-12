import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Institutional technical indicator engine.

    Guarantees:
    ✔ provider-agnostic column handling
    ✔ numeric stability
    ✔ zero divide protection
    ✔ deterministic output
    ✔ float32 safe
    """

    REQUIRED_COLUMN = "close"

    ####################################################
    # COLUMN NORMALIZATION
    ####################################################

    @staticmethod
    def _normalize_columns(df: pd.DataFrame):

        df = df.copy()

        df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            raise RuntimeError(
                "TechnicalIndicators requires 'close' column."
            )

        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

        if df["close"].isna().all():
            raise RuntimeError("Close column fully NaN.")

        df["close"].replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        df["close"].fillna(method="ffill", inplace=True)

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        return df

    ####################################################
    # MOVING AVERAGE
    ####################################################

    @classmethod
    def moving_average(cls, df: pd.DataFrame, window: int = 20):

        df = cls._normalize_columns(df)

        ma = df["close"].rolling(
            window=window,
            min_periods=window
        ).mean()

        return ma.astype("float32")

    ####################################################
    # RSI (INSTITUTIONAL SAFE)
    ####################################################

    @classmethod
    def rsi(cls, df: pd.DataFrame, window: int = 14):

        df = cls._normalize_columns(df)

        close = df["close"].astype("float64")

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(
            window=window,
            min_periods=window
        ).mean()

        avg_loss = loss.rolling(
            window=window,
            min_periods=window
        ).mean()

        rs = avg_gain / (avg_loss + 1e-12)

        rsi = 100 - (100 / (1 + rs))

        # Flat market → RSI 50
        flat_mask = (avg_gain == 0) & (avg_loss == 0)
        rsi.loc[flat_mask] = 50

        # Only gains → RSI 100
        gain_mask = (avg_gain > 0) & (avg_loss == 0)
        rsi.loc[gain_mask] = 100

        # Only losses → RSI 0
        loss_mask = (avg_gain == 0) & (avg_loss > 0)
        rsi.loc[loss_mask] = 0

        rsi.fillna(50, inplace=True)

        return rsi.clip(0, 100).astype("float32")

    ####################################################
    # BOLLINGER
    ####################################################

    @classmethod
    def bollinger_bands(cls, df: pd.DataFrame, window: int = 20):

        df = cls._normalize_columns(df)

        close = df["close"]

        ma = close.rolling(
            window=window,
            min_periods=window
        ).mean()

        std = close.rolling(
            window=window,
            min_periods=window
        ).std()

        upper = ma + 2 * std
        lower = ma - 2 * std

        return (
            upper.astype("float32"),
            lower.astype("float32")
        )

    ####################################################
    # MACD
    ####################################################

    @classmethod
    def macd(cls, df: pd.DataFrame):

        df = cls._normalize_columns(df)

        close = df["close"].astype("float64")

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()

        return (
            macd_line.astype("float32"),
            signal.astype("float32")
        )
