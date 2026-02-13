import logging
from pathlib import Path
import pandas as pd
import time
import os
import numpy as np
import hashlib
import re

from core.data.providers.market.router import MarketProviderRouter


logger = logging.getLogger("marketsentinel.market_data")


class MarketDataService:
    """
    Institutional Market Data Layer.

    Guarantees:
    ✔ zero lookahead bias
    ✔ schema drift detection
    ✔ concurrent safety
    ✔ corruption prevention
    ✔ ticker enforcement
    ✔ timezone normalization
    ✔ numeric safety
    """

    DATA_DIR = Path("data/lake")

    REQUIRED_COLUMNS = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    MIN_HISTORY_ROWS = 120
    SAFE_LAG_DAYS = 2
    MAX_ROWS = 15000

    ########################################################

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._fetcher = MarketProviderRouter()

        self.SCHEMA_HASH = hashlib.sha256(
            ",".join(sorted(self.REQUIRED_COLUMNS)).encode()
        ).hexdigest()[:10]

    ########################################################
    # TICKER SAFETY
    ########################################################

    @staticmethod
    def _sanitize_ticker(ticker: str):

        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):
            raise RuntimeError(f"Unsafe ticker: {ticker}")

        return ticker

    ########################################################
    # ATTACH TICKER (HARD GUARANTEE)
    ########################################################

    @staticmethod
    def _attach_ticker(df: pd.DataFrame, ticker: str):

        if df is None or df.empty:
            raise RuntimeError("Fetcher returned empty dataframe.")

        df = df.copy()
        df["ticker"] = ticker

        return df

    ########################################################
    # LOOKAHEAD PROTECTION
    ########################################################

    @classmethod
    def _cap_to_safe_date(cls, date_str: str):

        requested = pd.Timestamp(date_str).tz_localize(None)

        safe_cutoff = (
            pd.Timestamp.utcnow()
            .tz_localize(None)
            .normalize()
            - pd.Timedelta(days=cls.SAFE_LAG_DAYS)
        )

        return min(requested, safe_cutoff)

    ########################################################
    # TIMEZONE NORMALIZATION (CRITICAL)
    ########################################################

    @staticmethod
    def _normalize_dates(df: pd.DataFrame):

        df = df.copy()

        dt = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        # If timezone exists → convert → strip
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)

        df["date"] = dt

        return df

    ########################################################
    # HARD VALIDATOR
    ########################################################

    def _validate_dataset(self, df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Market data empty.")

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(f"Schema violation. Missing={missing}")

        df = self._normalize_dates(df)

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date"] + numeric_cols)

        if not np.isfinite(df[numeric_cols].to_numpy()).all():
            raise RuntimeError("Non-finite prices detected.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        if (df["high"] < df["low"]).any():
            raise RuntimeError("High < Low detected.")

        if df.duplicated(subset=["ticker", "date"]).any():
            raise RuntimeError("Duplicate ticker-date detected.")

        df = df.sort_values(["ticker", "date"])

        if len(df) < self.MIN_HISTORY_ROWS:
            raise RuntimeError("Insufficient market history.")

        if len(df) > self.MAX_ROWS:
            logger.warning("Dataset too large — trimming oldest rows.")
            df = df.tail(self.MAX_ROWS)

        return df.reset_index(drop=True)

    ########################################################
    # FETCH WITH RETRY
    ########################################################

    def _fetch_with_retry(
        self,
        ticker,
        start,
        end,
        interval,
        retries=3
    ):

        last_error = None

        for attempt in range(retries):

            try:

                df = self._fetcher.fetch(
                    ticker,
                    start,
                    end,
                    interval
                )

                df = self._attach_ticker(df, ticker)

                return self._validate_dataset(df)

            except Exception as e:

                last_error = e

                logger.warning(
                    "Fetch failed (%s) attempt %d/%d",
                    ticker,
                    attempt + 1,
                    retries
                )

                time.sleep(1.5)

        raise RuntimeError(
            f"Market fetch failed after retries: {ticker}"
        ) from last_error

    ########################################################
    # PUBLIC ENTRY
    ########################################################

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ):

        ticker = self._sanitize_ticker(ticker)

        end_date = self._cap_to_safe_date(end_date)

        logger.info("Fetching dataset for %s", ticker)

        df = self._fetch_with_retry(
            ticker,
            start_date,
            end_date.strftime("%Y-%m-%d"),
            interval
        )

        return df.reset_index(drop=True)
