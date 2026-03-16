import logging
import os
import random
import time
import threading
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_DAILY_INTERVALS = {"1d", "D", "1wk", "1mo"}
_INTRADAY_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "60m", "90m"}

_MAX_DAILY_GAP = 10


class StockPriceFetcher:
    """
    Wraps yfinance with retry logic, column normalisation, and
    basic data-quality guards.

    Designed to be called only by YahooProvider.
    """

    MAX_RETRIES = 5
    BASE_RETRY_SLEEP = 1.0
    MAX_BACKOFF = 10.0

    MIN_ROWS = 50
    REQUEST_TIMEOUT = 25

    SOFT_FAIL_MODE = os.getenv("YFINANCE_SOFT_MODE", "1") == "1"
    SOFT_FAIL_RATIO = 0.70

    # safer default interval between Yahoo requests
    MIN_REQUEST_INTERVAL = float(os.getenv("YFINANCE_MIN_INTERVAL", "2.0"))

    _last_request_time = 0.0
    _rate_lock = threading.Lock()

    # --------------------------------------------------
    # RATE LIMIT PROTECTION
    # --------------------------------------------------

    @classmethod
    def _respect_rate_limit(cls):

        with cls._rate_lock:

            now = time.time()

            elapsed = now - cls._last_request_time

            wait = cls.MIN_REQUEST_INTERVAL - elapsed

            if wait > 0:
                time.sleep(wait)

            cls._last_request_time = time.time()

    # --------------------------------------------------
    # COLUMN HELPERS
    # --------------------------------------------------

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:

        if isinstance(df.columns, pd.MultiIndex):

            normalized = []

            for col in df.columns:

                parts = [str(x).lower().strip() for x in col if str(x).strip()]

                canonical = None

                for field in ("adj close", "open", "high", "low", "close", "volume"):

                    if field in parts:
                        canonical = field.replace(" ", "_")
                        break

                normalized.append(canonical or parts[0] if parts else "unknown")

            df.columns = normalized

        else:

            df.columns = [
                str(c).lower().strip().replace(" ", "_")
                for c in df.columns
            ]

        if df.columns.duplicated().any():

            logger.warning("Duplicate columns after flattening — keeping first.")

            df = df.loc[:, ~df.columns.duplicated()]

        return df

    @staticmethod
    def _extract_date_column(df: pd.DataFrame) -> pd.DataFrame:

        if isinstance(df.index, pd.DatetimeIndex):

            date_values = df.index

            df = df.reset_index(drop=True)

            df["date"] = date_values

            return df

        df.columns = [str(c).lower().strip() for c in df.columns]

        for candidate in ("date", "datetime", "timestamp"):

            if candidate in df.columns:

                if candidate != "date":
                    df.rename(columns={candidate: "date"}, inplace=True)

                return df

        raise RuntimeError(
            "Could not locate a datetime column in Yahoo response. "
            f"Available columns: {list(df.columns)}"
        )

    @staticmethod
    def _ensure_utc(series: pd.Series) -> pd.Series:

        s = pd.to_datetime(series, errors="coerce")

        if s.isna().all():
            raise RuntimeError("Date column contains no valid datetimes.")

        if getattr(s.dt, "tz", None) is None:
            return s.dt.tz_localize("UTC")

        return s.dt.tz_convert("UTC")

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _validate_prices(df: pd.DataFrame) -> pd.DataFrame:

        for col in ("open", "high", "low", "close", "volume"):

            if col not in df.columns:
                raise RuntimeError(f"Missing required column: '{col}'")

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if df.empty:
            raise RuntimeError("All price rows are invalid after coercion.")

        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise RuntimeError("Non-positive prices detected in Yahoo data.")

        if (df["volume"] < 0).any():
            raise RuntimeError("Negative volume values detected.")

        return df

    # --------------------------------------------------
    # DOWNLOAD (UPDATED)
    # --------------------------------------------------

    def _download(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                self._respect_rate_limit()

                # safer endpoint than Ticker.history
                df = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False
                )

                if df is None or df.empty:
                    raise RuntimeError("yfinance returned empty DataFrame.")

                if len(df.columns) <= 1:
                    raise RuntimeError(
                        f"Suspicious yfinance response for {ticker}"
                    )

                logger.debug(
                    "yfinance download OK | ticker=%s attempt=%d rows=%d",
                    ticker,
                    attempt,
                    len(df),
                )

                return df

            except Exception as exc:

                last_exc = exc

                msg = str(exc).lower()

                backoff = min(
                    (2 ** (attempt - 1)) * self.BASE_RETRY_SLEEP
                    + random.uniform(0.5, 1.2),
                    self.MAX_BACKOFF,
                )

                if "too many requests" in msg or "rate" in msg:
                    backoff = max(backoff, 6.0)

                logger.warning(
                    "yfinance attempt %d/%d failed | ticker=%s | backoff=%.2fs | error=%s",
                    attempt,
                    self.MAX_RETRIES,
                    ticker,
                    backoff,
                    exc,
                )

                if attempt < self.MAX_RETRIES:
                    time.sleep(backoff)

        raise RuntimeError(
            f"yfinance fetch failed after {self.MAX_RETRIES} attempts "
            f"for '{ticker}'. Last error: {last_exc}"
        )

    # --------------------------------------------------
    # GAP CHECK
    # --------------------------------------------------

    @staticmethod
    def _check_calendar_gaps(df: pd.DataFrame, ticker: str, interval: str):

        if interval not in _DAILY_INTERVALS:
            return

        if len(df) < 2:
            return

        gap_days = df["date"].diff().dt.days

        max_gap = gap_days.max()

        if pd.notna(max_gap) and max_gap > _MAX_DAILY_GAP:

            logger.warning(
                "Large calendar gap for %s: %d days (interval=%s)",
                ticker,
                int(max_gap),
                interval,
            )

    # --------------------------------------------------
    # PUBLIC FETCH
    # --------------------------------------------------

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:

        df = self._download(ticker, start_date, end_date, interval)

        df = self._flatten_columns(df)

        df = self._extract_date_column(df)

        df["date"] = self._ensure_utc(df["date"])

        df.dropna(subset=["date"], inplace=True)

        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]

        required = {"open", "high", "low", "close", "volume"}

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Yahoo schema drift for '{ticker}' — missing: {missing}"
            )

        df = self._validate_prices(df)

        if df["date"].duplicated().any():

            n_dupes = df["date"].duplicated().sum()

            logger.warning(
                "Duplicate timestamps in Yahoo data for %s (%d rows)",
                ticker,
                n_dupes,
            )

            df = df.drop_duplicates(subset=["date"], keep="last")

        df = df.sort_values("date").reset_index(drop=True)

        now_utc = pd.Timestamp.now(tz="UTC")

        if df["date"].max() > now_utc:

            df = df[df["date"] <= now_utc].reset_index(drop=True)

        self._check_calendar_gaps(df, ticker, interval)

        if len(df) < self.MIN_ROWS:

            soft_threshold = int(self.MIN_ROWS * self.SOFT_FAIL_RATIO)

            if self.SOFT_FAIL_MODE and len(df) >= soft_threshold:

                logger.warning(
                    "Short history accepted in soft-fail mode for %s (%d rows).",
                    ticker,
                    len(df),
                )

            else:

                raise RuntimeError(
                    f"Insufficient history for '{ticker}': got {len(df)}"
                )

        df["ticker"] = ticker

        logger.info(
            "yfinance fetch success | ticker=%s interval=%s rows=%d",
            ticker,
            interval,
            len(df),
        )

        return df