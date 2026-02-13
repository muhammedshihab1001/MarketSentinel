import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Institutional technical indicator engine.

    Guarantees:
    ✔ No synthetic price creation
    ✔ Leak-safe rolling indicators
    ✔ Corruption detection
    ✔ Timestamp integrity
    ✔ Numeric safety
    """

    REQUIRED_COLUMN = "close"

    STD_FLOOR = 1e-6
    MAX_DAILY_RETURN = 0.80  # corruption guard

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

        ####################################################
        # DATE INTEGRITY
        ####################################################

        if "date" in df.columns:

            df["date"] = pd.to_datetime(
                df["date"],
                utc=True,
                errors="raise"
            )

            df = df.sort_values("date")

            if not df["date"].is_monotonic_increasing:
                raise RuntimeError(
                    "Non-monotonic timestamps detected."
                )

            if df["date"].duplicated().any():
                raise RuntimeError(
                    "Duplicate timestamps detected."
                )

        ####################################################
        # PRICE SAFETY
        ####################################################

        close = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

        close.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        if close.isna().any():
            raise RuntimeError(
                "Missing close prices detected — refusing indicator calc. "
                "Provider integrity compromised."
            )

        if (close <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        ####################################################
        # CORRUPTION GUARD
        ####################################################

        returns = close.pct_change().abs()

        if returns.dropna().max() > TechnicalIndicators.MAX_DAILY_RETURN:
            raise RuntimeError(
                "Unrealistic price jump detected — likely bad data."
            )

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

        ####################################################
        # EXPLOSION GUARD
        ####################################################

        macd_line = macd_line.clip(-500, 500)
        signal = signal.clip(-500, 500)

        return (
            macd_line.astype("float32"),
            signal.astype("float32")
        )
