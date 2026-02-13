import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Institutional technical indicator engine.
    """

    REQUIRED_COLUMN = "close"

    MAX_FORWARD_FILL = 2
    STD_FLOOR = 1e-6

    ####################################################
    # WINDOW VALIDATION
    ####################################################

    @staticmethod
    def _validate_window(window: int):
        if not isinstance(window, int):
            raise RuntimeError("Window must be int.")

        if window < 2:
            raise RuntimeError("Window must be >= 2.")

    ####################################################
    # COLUMN NORMALIZATION
    ####################################################

    @staticmethod
    def _normalize_columns(df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Indicator received empty dataframe.")

        df = df.copy()

        df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            raise RuntimeError(
                "TechnicalIndicators requires 'close' column."
            )

        if "date" in df.columns:
            df = df.sort_values("date")

            if df["date"].duplicated().any():
                raise RuntimeError("Duplicate timestamps detected.")

        close = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

        if close.isna().all():
            raise RuntimeError("Close column fully NaN.")

        close.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        close = close.ffill(limit=TechnicalIndicators.MAX_FORWARD_FILL)

        if close.isna().any():
            raise RuntimeError(
                "Close contains unresolved NaNs — refusing indicator calc."
            )

        if (close <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        df["close"] = close.astype("float32")

        return df

    ####################################################
    # MOVING AVERAGE
    ####################################################

    @classmethod
    def moving_average(cls, df: pd.DataFrame, window: int = 20):

        cls._validate_window(window)

        df = cls._normalize_columns(df)

        ma = df["close"].rolling(
            window=window,
            min_periods=window
        ).mean()

        return ma.astype("float32")

    ####################################################
    # RSI — WILDER SMOOTHING
    ####################################################

    @classmethod
    def rsi(cls, df: pd.DataFrame, window: int = 14):

        cls._validate_window(window)

        df = cls._normalize_columns(df)

        close = df["close"]

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1/window,
            adjust=False,
            min_periods=window
        ).mean()

        avg_loss = loss.ewm(
            alpha=1/window,
            adjust=False,
            min_periods=window
        ).mean()

        rs = avg_gain / (avg_loss + 1e-12)

        rsi = 100 - (100 / (1 + rs))

        rsi.fillna(50, inplace=True)

        return rsi.clip(0, 100).astype("float32")

    ####################################################
    # BOLLINGER
    ####################################################

    @classmethod
    def bollinger_bands(cls, df: pd.DataFrame, window: int = 20):

        cls._validate_window(window)

        df = cls._normalize_columns(df)

        close = df["close"]

        ma = close.rolling(
            window=window,
            min_periods=window
        ).mean()

        std = close.rolling(
            window=window,
            min_periods=window
        ).std(ddof=0)

        std = std.clip(lower=cls.STD_FLOOR)

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

        close = df["close"]

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()

        macd_line = macd_line.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)

        signal = signal.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)

        return (
            macd_line.astype("float32"),
            signal.astype("float32")
        )
