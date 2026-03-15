
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# Daily intervals — gap check is meaningful (> 10 calendar days = suspicious)
_DAILY_INTERVALS   = {"1d", "D", "1wk", "1mo"}
# Intraday intervals — multi-day gaps are normal (weekends / after-hours)
_INTRADAY_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "60m", "90m"}
# Max calendar gap (days) before a warning is emitted for daily data
_MAX_DAILY_GAP = 10


class StockPriceFetcher:
    """
    Wraps yfinance with retry logic, column normalisation, and
    basic data-quality guards.  Designed to be called only by
    YahooProvider — do not add business logic here.
    """

    # ── Retry settings ───────────────────────────────────────────────────────
    MAX_RETRIES      = 4
    BASE_RETRY_SLEEP = 1.0
    MAX_BACKOFF      = 6.0

    # ── Hard limits ──────────────────────────────────────────────────────────
    MIN_ROWS         = 50          # absolute floor before soft/hard fail
    REQUEST_TIMEOUT  = 20          # seconds passed to yfinance

    # ── Soft-fail: accept short history in demo / CI ─────────────────────────
    SOFT_FAIL_MODE   = os.getenv("YFINANCE_SOFT_MODE", "1") == "1"
    SOFT_FAIL_RATIO  = 0.70

    # ────────────────────────────────────────────────────────────────────────
    # COLUMN HELPERS
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten MultiIndex columns produced by yfinance and
        lowercase + strip all column names.

        MultiIndex example:
            ('Open', 'AAPL') → 'open'
            ('Adj Close', '') → 'adj_close'
        """
        if isinstance(df.columns, pd.MultiIndex):
            normalized = []
            for col in df.columns:
                parts = [str(x).lower().strip() for x in col if str(x).strip()]
                # Map known OHLCV fields first so we get clean canonical names
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

        # Drop duplicate columns — keep first occurrence
        if df.columns.duplicated().any():
            logger.warning("Duplicate columns after flattening — keeping first.")
            df = df.loc[:, ~df.columns.duplicated()]

        return df

    @staticmethod
    def _extract_date_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Promote the DatetimeIndex (or a date-named column) to a regular
        'date' column.  Raises if no datetime source can be found.
        """
        # Case 1: index is already a DatetimeIndex — most common yfinance case
        if isinstance(df.index, pd.DatetimeIndex):
            date_values = df.index
            df = df.reset_index(drop=True)
            df["date"] = date_values
            return df

        # Case 2: index was reset and ended up as a column
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
        """Parse a Series to UTC-aware datetime."""
        s = pd.to_datetime(series, errors="coerce")
        if s.isna().all():
            raise RuntimeError("Date column contains no valid datetimes.")
        if getattr(s.dt, "tz", None) is None:
            return s.dt.tz_localize("UTC")
        return s.dt.tz_convert("UTC")

    # ────────────────────────────────────────────────────────────────────────
    # VALIDATION  (schema + price sanity only — no repair logic)
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_prices(df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce numeric types, drop NaN price rows, and check for
        obviously bad values (non-positive prices, negative volume).
        OHLC repair is intentionally left to YahooProvider.
        """
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

    # ────────────────────────────────────────────────────────────────────────
    # DOWNLOAD WITH EXPONENTIAL BACKOFF
    # ────────────────────────────────────────────────────────────────────────

    def _download(
        self,
        ticker:   str,
        start:    str,
        end:      str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Call yfinance with exponential backoff + jitter.
        Uses a fresh Ticker object per attempt to avoid session-level
        throttle coupling.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,      # keep raw + adj_close separate
                    timeout=self.REQUEST_TIMEOUT,
                )

                if df is None or df.empty:
                    raise RuntimeError("yfinance returned an empty DataFrame.")

                if len(df.columns) <= 1:
                    raise RuntimeError(
                        f"Suspicious yfinance response for {ticker} "
                        f"— only {len(df.columns)} column(s) returned."
                    )

                logger.debug(
                    "yfinance download OK | ticker=%s attempt=%d rows=%d",
                    ticker, attempt, len(df),
                )
                return df

            except Exception as exc:
                last_exc = exc
                backoff = min(
                    (2 ** (attempt - 1)) * self.BASE_RETRY_SLEEP
                    + random.uniform(0.2, 0.8),
                    self.MAX_BACKOFF,
                )
                logger.warning(
                    "yfinance attempt %d/%d failed | ticker=%s | "
                    "backoff=%.2fs | error=%s",
                    attempt, self.MAX_RETRIES, ticker, backoff, exc,
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(backoff)

        raise RuntimeError(
            f"yfinance fetch failed after {self.MAX_RETRIES} attempts "
            f"for '{ticker}'. Last error: {last_exc}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # CALENDAR GAP CHECK  (interval-aware)
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_calendar_gaps(df: pd.DataFrame, ticker: str, interval: str) -> None:
        """
        Warn if there are unexpectedly large gaps in daily data.
        Skipped entirely for intraday intervals where multi-day gaps
        are normal (weekends, holidays, after-hours sessions).
        """
        if interval not in _DAILY_INTERVALS:
            return   # intraday gaps are expected — nothing to check

        if len(df) < 2:
            return

        gap_days = df["date"].diff().dt.days
        max_gap  = gap_days.max()

        if pd.notna(max_gap) and max_gap > _MAX_DAILY_GAP:
            logger.warning(
                "Large calendar gap for %s: %d days (interval=%s). "
                "Possible data outage or delisting period.",
                ticker, int(max_gap), interval,
            )

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC FETCH
    # ────────────────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker:     str,
        start_date: str,
        end_date:   str,
        interval:   str = "1d",
    ) -> pd.DataFrame:
        """
        Download and lightly clean OHLCV data from Yahoo Finance.

        Returns a DataFrame with columns:
            date (UTC-aware), open, high, low, close, volume, ticker

        Parameters
        ----------
        ticker     : Stock symbol, e.g. "AAPL"
        start_date : ISO date string, e.g. "2023-01-01"
        end_date   : ISO date string, e.g. "2024-01-01"
        interval   : yfinance interval string, e.g. "1d", "1h", "15m"
        """

        # ── 1. Download ──────────────────────────────────────────────────────
        df = self._download(ticker, start_date, end_date, interval)

        # ── 2. Flatten columns ───────────────────────────────────────────────
        df = self._flatten_columns(df)

        # ── 3. Extract date ──────────────────────────────────────────────────
        df = self._extract_date_column(df)
        df["date"] = self._ensure_utc(df["date"])
        df.dropna(subset=["date"], inplace=True)

        # ── 4. Prefer adj_close over close (splits / dividends) ──────────────
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]
            logger.debug("Using adj_close as close for %s.", ticker)

        # ── 5. Schema guard ──────────────────────────────────────────────────
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Yahoo schema drift for '{ticker}' — missing: {missing}"
            )

        # ── 6. Price / volume validation ─────────────────────────────────────
        df = self._validate_prices(df)

        # ── 7. Deduplicate timestamps (raise→dedupe: yfinance can return dupes)
        if df["date"].duplicated().any():
            n_dupes = df["date"].duplicated().sum()
            logger.warning(
                "Duplicate timestamps in Yahoo data for %s (%d rows) "
                "— keeping last occurrence.",
                ticker, n_dupes,
            )
            df = df.drop_duplicates(subset=["date"], keep="last")

        # ── 8. Sort chronologically ──────────────────────────────────────────
        df = df.sort_values("date").reset_index(drop=True)

        # ── 9. Future-candle guard ───────────────────────────────────────────
        now_utc = pd.Timestamp.now(tz="UTC")
        if df["date"].max() > now_utc:
            logger.warning(
                "Future candle detected for %s — stripping %d row(s).",
                ticker,
                (df["date"] > now_utc).sum(),
            )
            df = df[df["date"] <= now_utc].reset_index(drop=True)

        # ── 10. Calendar gap check (daily only) ──────────────────────────────
        self._check_calendar_gaps(df, ticker, interval)

        # ── 11. Min-rows gate ─────────────────────────────────────────────────
        if len(df) < self.MIN_ROWS:
            soft_threshold = int(self.MIN_ROWS * self.SOFT_FAIL_RATIO)
            if self.SOFT_FAIL_MODE and len(df) >= soft_threshold:
                logger.warning(
                    "Short history accepted in soft-fail mode for %s "
                    "(%d rows, floor=%d).",
                    ticker, len(df), self.MIN_ROWS,
                )
            else:
                raise RuntimeError(
                    f"Insufficient history for '{ticker}': "
                    f"got {len(df)}, need {self.MIN_ROWS} "
                    f"(soft_fail={'on' if self.SOFT_FAIL_MODE else 'off'})"
                )

        # ── 12. Tag with ticker ───────────────────────────────────────────────
        df["ticker"] = ticker

        logger.info(
            "yfinance fetch success | ticker=%s interval=%s rows=%d",
            ticker, interval, len(df),
        )

        return df