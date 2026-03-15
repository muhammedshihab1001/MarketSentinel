
import logging
import os
import threading
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from core.data.providers.market.base import MarketDataProvider

logger = logging.getLogger(__name__)

# TwelveData free tier: 8 calls per 60-second rolling window
_TD_MAX_CALLS    = 8
_TD_WINDOW_SECS  = 60

# Max bars requested per call — TD free tier cap
_DEFAULT_OUTPUTSIZE = 5000


class TwelveDataProvider(MarketDataProvider):
    """
    Twelve Data daily + intraday OHLCV provider.
    Used as fallback #2 when Yahoo Finance and AlphaVantage are unavailable.
    """

    PROVIDER_NAME = "twelvedata"

    BASE_URL = "https://api.twelvedata.com/time_series"

    # ── Retry settings ───────────────────────────────────────────────────────
    MAX_RETRIES = 3
    RETRY_SLEEP = 1.5      # base seconds; multiplied by attempt number

    # ── Row limits ───────────────────────────────────────────────────────────
    DEFAULT_MIN_ROWS  = 120
    MAX_OUTPUTSIZE    = _DEFAULT_OUTPUTSIZE

    # ── Rate limiter (class-level sliding window) ─────────────────────────────
    _CALL_LOCK       = threading.Lock()
    _CALL_TIMESTAMPS: List[float] = []
    _MAX_CALLS       = _TD_MAX_CALLS
    _WINDOW_SECS     = _TD_WINDOW_SECS

    # ── Interval alias map: canonical → TwelveData format ────────────────────
    INTERVAL_MAP = {
        "1d":  "1day",
        "D":   "1day",
        "1h":  "1h",
        "60m": "1h",
        "15m": "15min",
        "5m":  "5min",
        "1m":  "1min",
    }

    # ────────────────────────────────────────────────────────────────────────
    # INIT
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.api_key: str = os.getenv("TWELVEDATA_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "TWELVEDATA_API_KEY is not set. "
                "TwelveData provider cannot be initialised."
            )
        self.session = requests.Session()
        logger.info("TwelveDataProvider ready (rate-aware).")

    # ────────────────────────────────────────────────────────────────────────
    # RATE LIMITER  (sliding window)
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def _respect_rate_limit(cls) -> None:
        """
        Enforce TD's 8-calls/minute free-tier limit using a
        sliding window. Thread-safe via class-level lock.
        """
        with cls._CALL_LOCK:
            now = time.time()

            # Evict timestamps outside the current window
            cls._CALL_TIMESTAMPS = [
                t for t in cls._CALL_TIMESTAMPS
                if now - t < cls._WINDOW_SECS
            ]

            if len(cls._CALL_TIMESTAMPS) >= cls._MAX_CALLS:
                # Wait until the oldest call ages out of the window
                sleep_for = (cls._WINDOW_SECS - (now - cls._CALL_TIMESTAMPS[0])) + 0.25
                if sleep_for > 0:
                    logger.debug(
                        "TwelveData rate-limit pause: %.2fs", sleep_for
                    )
                    time.sleep(sleep_for)

            cls._CALL_TIMESTAMPS.append(time.time())

    # ────────────────────────────────────────────────────────────────────────
    # API CALL
    # ────────────────────────────────────────────────────────────────────────

    def _call_api(self, params: dict) -> dict:
        """
        Call TwelveData with linear backoff retry.
        Handles vendor error codes without double-sleeping.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._respect_rate_limit()

                r = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=(5, 15),
                )
                r.raise_for_status()
                data = r.json()

                # ── TD vendor error responses ─────────────────────────────────
                if "code" in data:
                    msg = data.get("message", "Unknown TwelveData error")
                    # Rate-limit errors get an extra note in the log,
                    # but the retry loop + _respect_rate_limit handles
                    # the actual delay — no extra sleep here.
                    if "frequency" in msg.lower() or "limit" in msg.lower():
                        logger.warning(
                            "TwelveData rate-limit response "
                            "(attempt %d/%d): %s",
                            attempt, self.MAX_RETRIES, msg,
                        )
                    raise RuntimeError(f"TwelveData API error: {msg}")

                return data

            except Exception as exc:
                logger.warning(
                    "TwelveData attempt %d/%d failed | error=%s",
                    attempt, self.MAX_RETRIES, exc,
                )
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"TwelveData fetch failed after {self.MAX_RETRIES} "
                        f"attempts. Last error: {exc}"
                    ) from exc
                time.sleep(self.RETRY_SLEEP * attempt)

    # ────────────────────────────────────────────────────────────────────────
    # NORMALISER
    # ────────────────────────────────────────────────────────────────────────

    def _normalize(
        self,
        values:     list,
        ticker:     str,
        start_date: str,
        end_date:   str,
        min_rows:   int,
    ) -> pd.DataFrame:
        """
        Parse TwelveData 'values' list → clean OHLCV DataFrame.
        Applies client-side date range filter as a safety net.
        """

        df = pd.DataFrame(values)

        if df.empty:
            raise RuntimeError(
                f"TwelveData returned empty dataset for '{ticker}'."
            )

        # ── Rename datetime → date ────────────────────────────────────────────
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        if "date" not in df.columns:
            raise RuntimeError(
                f"TwelveData response missing 'datetime' field for '{ticker}'."
            )

        # ── Date parsing + client-side range filter ───────────────────────────
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts   = pd.Timestamp(end_date,   tz="UTC")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

        if df.empty:
            raise RuntimeError(
                f"TwelveData returned no data for '{ticker}' "
                f"in range [{start_date}, {end_date}]."
            )

        # ── Schema guard ──────────────────────────────────────────────────────
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"TwelveData schema violation for '{ticker}': "
                f"missing columns {missing}."
            )

        # ── Numeric coercion ──────────────────────────────────────────────────
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if df.empty:
            raise RuntimeError(
                f"All rows invalid after coercion for '{ticker}'."
            )

        # ── Basic price guards ────────────────────────────────────────────────
        df = df[df["high"] >= df["low"]]
        df = df[df["close"] > 0]

        if df.empty:
            raise RuntimeError(
                f"No valid OHLC bars remaining for '{ticker}' after price guards."
            )

        # ── Extreme-move check — warn only (router handles rejection) ─────────
        # Hard-raising here would prevent the router from falling back cleanly.
        jumps = df["close"].pct_change().abs()
        if jumps.dropna().max() > 0.90:
            logger.warning(
                "TwelveData: extreme price move detected for %s — "
                "router will validate further.",
                ticker,
            )

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
                f"TwelveData insufficient history for '{ticker}': "
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
        Fetch OHLCV data from TwelveData.

        Parameters
        ----------
        ticker     : Stock symbol, e.g. "AAPL"
        start_date : ISO string, e.g. "2023-01-01"
        end_date   : ISO string, e.g. "2024-01-01"
        interval   : Any canonical interval (see INTERVAL_MAP for supported values)
        min_rows   : Passed via kwargs; defaults to DEFAULT_MIN_ROWS
        """

        # ── Interval validation + mapping ─────────────────────────────────────
        # Validate BEFORE mapping so the error message shows the original value
        if interval not in self.INTERVAL_MAP:
            raise ValueError(
                f"TwelveData: unsupported interval '{interval}'. "
                f"Allowed: {sorted(self.INTERVAL_MAP.keys())}"
            )
        td_interval = self.INTERVAL_MAP[interval]   # keep original untouched

        min_rows: int = kwargs.get("min_rows", self.DEFAULT_MIN_ROWS)

        params = {
            "symbol":     ticker,
            "interval":   td_interval,
            "start_date": start_date,
            "end_date":   end_date,
            "outputsize": self.MAX_OUTPUTSIZE,
            "apikey":     self.api_key,
            "format":     "JSON",
        }

        data = self._call_api(params)

        if "values" not in data:
            raise RuntimeError(
                f"TwelveData response missing 'values' key for '{ticker}'. "
                f"Possible schema change. Keys present: {list(data.keys())}"
            )

        df = self._normalize(
            values=data["values"],
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            min_rows=min_rows,
        )

        logger.info(
            "TwelveData served | ticker=%s interval=%s rows=%d range=[%s, %s]",
            ticker, interval, len(df), start_date, end_date,
        )

        return self.validate_contract(df)