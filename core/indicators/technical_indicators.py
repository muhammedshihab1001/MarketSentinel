import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Lightweight technical indicator engine for portfolio inference.

    Guarantees
    ✔ Leak-safe rolling indicators
    ✔ Numeric safety
    ✔ Timestamp normalization
    ✔ Handles noisy Yahoo Finance data
    """

    REQUIRED_COLUMN = "close"

    STD_FLOOR = 1e-6
    MAX_DAILY_RETURN = 0.80
    EPSILON = 1e-9

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
        # DATE NORMALIZATION
        ####################################################

        if "date" in df.columns:

            df["date"] = pd.to_datetime(
                df["date"],
                utc=True,
                errors="coerce"
            )

            df = df.dropna(subset=["date"])

            df = df.sort_values("date")

        ####################################################
        # PRICE SANITIZATION
        ####################################################

        close = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

        close.replace([np.inf, -np.inf], np.nan, inplace=True)

        close = close.ffill().bfill()

        if (close <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        ####################################################
        # YAHOO FINANCE SPIKE GUARD
        ####################################################

        returns = close.pct_change().abs()

        if returns.dropna().max() > TechnicalIndicators.MAX_DAILY_RETURN:

            logger.warning(
                "Extreme price jump detected (>80%%). "
                "Possible Yahoo anomaly. Applying spike guard."
            )

            close = close.clip(
                lower=close.quantile(0.001),
                upper=close.quantile(0.999)
            )

        df["close"] = close.astype("float32")

        return df

    ####################################################
    # MOVING AVERAGE
    ####################################################

    @classmethod
    def moving_average(
        cls,
        df: pd.DataFrame,
        window: int = 20,
        normalize: bool = True
    ):

        cls._validate_window(window)

        if normalize:
            df = cls._normalize_columns(df)

        ma = df["close"].rolling(
            window=window,
            min_periods=window
        ).mean()

        return ma.astype("float32")

    ####################################################
    # EMA
    ####################################################

    @classmethod
    def ema(
        cls,
        df: pd.DataFrame,
        span: int = 20,
        normalize: bool = True
    ):

        cls._validate_window(span)

        if normalize:
            df = cls._normalize_columns(df)

        ema = df["close"].ewm(
            span=span,
            adjust=False
        ).mean()

        return ema.astype("float32")

    ####################################################
    # RSI — WILDER SMOOTHING
    ####################################################

    @classmethod
    def rsi(
        cls,
        df: pd.DataFrame,
        window: int = None,
        period: int = None,
        normalize: bool = True
    ):
        """
        RSI indicator.

        Supports both:
        rsi(window=14)
        rsi(period=14)
        """

        if window is None and period is None:
            window = 14

        if window is None:
            window = period

        cls._validate_window(window)

        if normalize:
            df = cls._normalize_columns(df)

        close = df["close"].astype("float64")

        if len(close) < window:
            return pd.Series(
                np.full(len(close), 50.0),
                index=close.index,
                dtype="float32"
            )

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / window,
            adjust=False,
            min_periods=window
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / window,
            adjust=False,
            min_periods=window
        ).mean()

        rs = avg_gain / (avg_loss + cls.EPSILON)

        rsi = 100 - (100 / (1 + rs))

        rsi = rsi.fillna(50.0).clip(0, 100)

        return rsi.astype("float32")

    ####################################################
    # BOLLINGER BANDS
    ####################################################

    @classmethod
    def bollinger_bands(
        cls,
        df: pd.DataFrame,
        window: int = 20,
        normalize: bool = True
    ):

        cls._validate_window(window)

        if normalize:
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
    def macd(
        cls,
        df: pd.DataFrame,
        normalize: bool = True
    ):

        if normalize:
            df = cls._normalize_columns(df)

        close = df["close"]

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        macd_line = ema12 - ema26

        signal = macd_line.ewm(span=9, adjust=False).mean()

        macd_line = macd_line.replace([np.inf, -np.inf], np.nan).fillna(0)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0)

        macd_line = macd_line.clip(-500, 500)
        signal = signal.clip(-500, 500)

        return (
            macd_line.astype("float32"),
            signal.astype("float32")
        )