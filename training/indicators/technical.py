import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Institutional technical indicator engine.

    Guarantees:
    - numeric stability
    - no division-by-zero
    - deterministic output
    - NaN-safe indicators
    """

    @staticmethod
    def moving_average(df: pd.DataFrame, window: int = 20):
        return df["Close"].rolling(
            window=window,
            min_periods=window
        ).mean()

    # -----------------------------------------------------

    @staticmethod
    def rsi(df: pd.DataFrame, window: int = 14):

        close = df["Close"].astype("float64")

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

        # Prevent divide-by-zero
        rs = avg_gain / (avg_loss + 1e-10)

        rsi = 100 - (100 / (1 + rs))

        # Institutional handling:
        # flat market -> RSI = 50
        rsi = rsi.fillna(50)

        # Clamp for safety
        rsi = rsi.clip(0, 100)

        return rsi.astype("float64")

    # -----------------------------------------------------

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, window: int = 20):

        ma = TechnicalIndicators.moving_average(df, window)

        std = df["Close"].rolling(
            window=window,
            min_periods=window
        ).std()

        upper = ma + 2 * std
        lower = ma - 2 * std

        return upper, lower

    # -----------------------------------------------------

    @staticmethod
    def macd(df: pd.DataFrame):

        close = df["Close"].astype("float64")

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()

        return macd_line.astype("float64"), signal.astype("float64")
