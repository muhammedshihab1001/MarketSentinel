
import logging
import os
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from core.data.providers.market.base import MarketDataProvider

logger = logging.getLogger(__name__)

# AV free-tier: 5 calls per minute → 1 call per 12.5s minimum
_AV_FREE_TIER_INTERVAL = 12.5

# If date range is within this many days, use compact (100 bars) not full
_COMPACT_THRESHOLD_DAYS = 90


class AlphaVantageProvider(MarketDataProvider):
    """
    Alpha Vantage daily OHLCV provider.
    Used as fallback #1 when Yahoo Finance is unavailable.
    Supports daily interval only.
    """

    PROVIDER_NAME = "alphavantage"

    BASE_URL = "https://www.alphavantage.co/query"

    # ── Retry settings ───────────────────────────────────────────────────────
    MAX_RETRIES  = 3
    RETRY_SLEEP  = 12          # seconds between retries

    # ── Row limits ───────────────────────────────────────────────────────────
    DEFAULT_MIN_ROWS = 120

    # ── Rate limiter (class-level — shared across all instances) ─────────────
    _CALL_LOCK    = threading.Lock()
    _LAST_CALL    = 0.0
    _MIN_INTERVAL = _AV_FREE_TIER_INTERVAL

    # ── Supported intervals (daily only) ─────────────────────────────────────
    ALLOWED_INTERVALS = {"1d", "D"}

    # ────────────────────────────────────────────────────────────────────────
    # INIT
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.api_key: str = os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "ALPHAVANTAGE_API_KEY is not set. "
                "AlphaVantage provider cannot be initialised."
            )
        self.session = requests.Session()
        logger.info("AlphaVantageProvider ready.")

    # ────────────────────────────────────────────────────────────────────────
    # RATE LIMITER
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def _respect_rate_limit(cls) -> None:
        """
        Enforce AV's 5-calls/minute free-tier limit.
        Uses a class-level lock so all instances share the same counter.
        """
        with cls._CALL_LOCK:
            elapsed = time.time() - cls._LAST_CALL
            if elapsed < cls._MIN_INTERVAL:
                wait = cls._MIN_INTERVAL - elapsed
                logger.debug("AlphaVantage rate-limit pause: %.2fs", wait)
                time.sleep(wait)
            cls._LAST_CALL = time.time()

    # ────────────────────────────────────────────────────────────────────────
    # OUTPUT SIZE HELPER
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_outputsize(start_date: str) -> str:
        """
        Use 'compact' (100 bars) for recent date ranges,
        'full' (20 years) only when the start date is old.
        Avoids downloading 20 years of data for a 30-day request.
        """
        try:
            start_ts = pd.Timestamp(start_date, tz="UTC")
            days_back = (pd.Timestamp.now(tz="UTC") - start_ts).days
            return "compact" if days_back <= _COMPACT_THRESHOLD_DAYS else "full"
        except Exception:
            return "full"   # safe default on parse failure

    # ────────────────────────────────────────────────────────────────────────
    # API CALL
    # ────────────────────────────────────────────────────────────────────────

    def _call_api(self, symbol: str, outputsize: str = "full") -> dict:
        """
        Call AV TIME_SERIES_DAILY_ADJUSTED with exponential-ish backoff.
        Handles rate-limit "Note" responses without double-sleeping.
        """
        params = {
            "function":   "TIME_SERIES_DAILY_ADJUSTED",
            "symbol":     symbol,
            "outputsize": outputsize,
            "apikey":     self.api_key,
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._respect_rate_limit()

                r = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=(6, 20),
                )
                r.raise_for_status()
                data = r.json()

                # ── AV error responses ────────────────────────────────────────
                if "Error Message" in data:
                    raise RuntimeError(
                        f"AlphaVantage error for '{symbol}': {data['Error Message']}"
                    )

                if "Information" in data:
                    # Premium endpoint hit on free key
                    raise RuntimeError(
                        f"AlphaVantage plan restriction for '{symbol}': "
                        f"{data['Information']}"
                    )

                if "Note" in data:
                    # Rate-limit notice — _respect_rate_limit will handle the
                    # delay on the next attempt; no extra sleep needed here.
                    logger.warning(
                        "AlphaVantage rate-limit note received "
                        "(attempt %d/%d) — will retry.",
                        attempt, self.MAX_RETRIES,
                    )
                    raise RuntimeError("AlphaVantage rate limited.")

                return data

            except Exception as exc:
                logger.warning(
                    "AlphaVantage attempt %d/%d failed | ticker=%s | error=%s",
                    attempt, self.MAX_RETRIES, symbol, exc,
                )
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"AlphaVantage fetch failed after {self.MAX_RETRIES} "
                        f"attempts for '{symbol}'. Last error: {exc}"
                    ) from exc
                time.sleep(self.RETRY_SLEEP)

    # ────────────────────────────────────────────────────────────────────────
    # NORMALISER
    # ────────────────────────────────────────────────────────────────────────

    def _normalize(
        self,
        data:       dict,
        ticker:     str,
        start_date: str,
        end_date:   str,
        min_rows:   int,
    ) -> pd.DataFrame:
        """
        Parse AV JSON → clean OHLCV DataFrame filtered to [start_date, end_date].
        Uses adjusted close for consistency with Yahoo primary source.
        """

        key = "Time Series (Daily)"
        if key not in data:
            raise RuntimeError(
                "AlphaVantage response schema changed — "
                f"expected key '{key}' not found."
            )

        df = (
            pd.DataFrame.from_dict(data[key], orient="index")
            .reset_index()
            .rename(columns={"index": "date"})
        )

        # ── Map AV field names → canonical names ─────────────────────────────
        # Use adjusted close ("5. adjusted close") to match Yahoo's adj_close,
        # keeping data consistent when falling back from primary source.
        rename_map = {
            "1. open":             "open",
            "2. high":             "high",
            "3. low":              "low",
            "5. adjusted close":   "close",   # adjusted — not raw "4. close"
            "6. volume":           "volume",
        }
        df.rename(columns=rename_map, inplace=True)

        # ── Schema guard ──────────────────────────────────────────────────────
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"AlphaVantage schema violation for '{ticker}': "
                f"missing columns {missing}. "
                f"Available: {list(df.columns)}"
            )

        # ── Date parsing + range filter ───────────────────────────────────────
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts   = pd.Timestamp(end_date,   tz="UTC")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

        if df.empty:
            raise RuntimeError(
                f"AlphaVantage returned no data for '{ticker}' "
                f"in range [{start_date}, {end_date}]."
            )

        # ── Numeric coercion ──────────────────────────────────────────────────
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        # ── Basic OHLC sanity ─────────────────────────────────────────────────
        df = df[df["high"] >= df["low"]]
        df = df[df["close"] > 0]

        # ── Sort + deduplicate ────────────────────────────────────────────────
        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        # ── Min-rows gate ─────────────────────────────────────────────────────
        if len(df) < min_rows:
            raise RuntimeError(
                f"AlphaVantage insufficient history for '{ticker}': "
                f"got {len(df)}, need {min_rows}."
            )

        df["ticker"] = ticker
        return df

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC FETCH  (called by MarketProviderRouter)
    # ────────────────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker:     str,
        start_date: str,
        end_date:   str,
        interval:   str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data from Alpha Vantage.

        Parameters
        ----------
        ticker     : Stock symbol, e.g. "AAPL"
        start_date : ISO string, e.g. "2023-01-01"
        end_date   : ISO string, e.g. "2024-01-01"
        interval   : Must be "1d" or "D" (daily only)
        min_rows   : Passed via kwargs; defaults to DEFAULT_MIN_ROWS
        """

        # ── Interval guard ────────────────────────────────────────────────────
        if interval not in self.ALLOWED_INTERVALS:
            raise ValueError(
                f"AlphaVantage supports daily intervals only. "
                f"Got: '{interval}'. Allowed: {self.ALLOWED_INTERVALS}"
            )

        min_rows: int = kwargs.get("min_rows", self.DEFAULT_MIN_ROWS)

        # ── Auto-select outputsize to avoid 20-year downloads ─────────────────
        outputsize = self._pick_outputsize(start_date)
        logger.debug(
            "AlphaVantage outputsize=%s for ticker=%s start=%s",
            outputsize, ticker, start_date,
        )

        # ── Fetch + normalise ─────────────────────────────────────────────────
        data = self._call_api(ticker, outputsize=outputsize)

        df = self._normalize(
            data=data,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            min_rows=min_rows,
        )

        logger.info(
            "AlphaVantage served | ticker=%s rows=%d range=[%s, %s]",
            ticker, len(df), start_date, end_date,
        )

        return self.validate_contract(df)