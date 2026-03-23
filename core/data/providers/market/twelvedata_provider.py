import logging
import os
import threading
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from core.data.providers.market.base import MarketDataProvider

logger = logging.getLogger(__name__)

_TD_MAX_CALLS = 8
_TD_WINDOW_SECS = 60
_DEFAULT_OUTPUTSIZE = 5000

# FIX: Delta window threshold — same logic as yahoo_provider.
# If the requested date window is <= 30 days, skip the min_rows check.
DELTA_WINDOW_DAYS = 30


class TwelveDataProvider(MarketDataProvider):
    """
    TwelveData OHLCV provider.
    Used as fallback when Yahoo Finance fails.
    """

    PROVIDER_NAME = "twelvedata"

    BASE_URL = "https://api.twelvedata.com/time_series"

    MAX_RETRIES = 3
    RETRY_SLEEP = 1.5

    DEFAULT_MIN_ROWS = 120
    MAX_OUTPUTSIZE = _DEFAULT_OUTPUTSIZE

    _CALL_LOCK = threading.Lock()
    _CALL_TIMESTAMPS: List[float] = []
    _MAX_CALLS = _TD_MAX_CALLS
    _WINDOW_SECS = _TD_WINDOW_SECS

    INTERVAL_MAP = {
        "1d": "1day",
        "D": "1day",
        "1h": "1h",
        "60m": "1h",
        "15m": "15min",
        "5m": "5min",
        "1m": "1min",
    }

    def __init__(self) -> None:

        self.api_key: str = os.getenv("TWELVEDATA_API_KEY", "")

        if not self.api_key:
            raise RuntimeError(
                "TWELVEDATA_API_KEY is not set. TwelveData provider cannot initialise."
            )

        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)

        logger.info("TwelveDataProvider ready (rate-aware).")

    ############################################################
    # DELTA WINDOW DETECTION
    # FIX: Same pattern as yahoo_provider — skip min_rows for
    # short date windows (daily sync requests only 5-14 days).
    ############################################################

    @staticmethod
    def _is_delta_fetch(start_date: str, end_date: str) -> bool:
        try:
            start = datetime.strptime(start_date[:10], "%Y-%m-%d")
            end = datetime.strptime(end_date[:10], "%Y-%m-%d")
            return (end - start).days <= DELTA_WINDOW_DAYS
        except Exception:
            return False

    ############################################################
    # RATE LIMITER
    #
    # FIX: The original code slept INSIDE the lock, which caused
    # all worker threads to queue behind the sleeping thread.
    # With workers=4 all hitting the rate limit simultaneously,
    # this produced 50-58s stalls for every burst.
    #
    # Fix: calculate sleep time inside the lock, release the lock,
    # then sleep, then re-acquire and record the timestamp.
    # Each thread now sleeps independently without blocking others.
    ############################################################

    @classmethod
    def _respect_rate_limit(cls) -> None:

        sleep_for = 0.0

        with cls._CALL_LOCK:
            now = time.time()

            # Evict timestamps older than the window
            cls._CALL_TIMESTAMPS = [
                t for t in cls._CALL_TIMESTAMPS
                if now - t < cls._WINDOW_SECS
            ]

            if len(cls._CALL_TIMESTAMPS) >= cls._MAX_CALLS:
                # Calculate required sleep but DO NOT sleep here
                sleep_for = cls._WINDOW_SECS - (now - cls._CALL_TIMESTAMPS[0]) + 0.5
                sleep_for = max(sleep_for, 0.0)

        # FIX: Sleep OUTSIDE the lock so other threads are not blocked
        if sleep_for > 0:
            logger.debug("TwelveData rate-limit pause: %.2fs", sleep_for)
            time.sleep(sleep_for)

        # Re-acquire and record the timestamp after sleeping
        with cls._CALL_LOCK:
            cls._CALL_TIMESTAMPS.append(time.time())

    def _call_api(self, params: dict) -> dict:

        for attempt in range(1, self.MAX_RETRIES + 1):

            start_time = time.time()

            try:

                self._respect_rate_limit()

                r = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=(5, 15),
                )

                latency = time.time() - start_time

                if latency > 8:
                    logger.warning(
                        "Slow TwelveData API response | latency=%.2fs",
                        latency,
                    )

                r.raise_for_status()

                data = r.json()

                if "code" in data:
                    msg = data.get("message", "Unknown TwelveData error")

                    if "frequency" in msg.lower() or "limit" in msg.lower():
                        logger.warning(
                            "TwelveData rate-limit response (attempt %d/%d): %s",
                            attempt,
                            self.MAX_RETRIES,
                            msg,
                        )

                    raise RuntimeError(f"TwelveData API error: {msg}")

                return data

            except Exception as exc:

                logger.warning(
                    "TwelveData attempt %d/%d failed | error=%s",
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                )

                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"TwelveData fetch failed after {self.MAX_RETRIES} attempts. "
                        f"Last error: {exc}"
                    ) from exc

                time.sleep(self.RETRY_SLEEP * attempt)

    ############################################################
    # NORMALIZE
    ############################################################

    def _normalize(
        self,
        values: list,
        ticker: str,
        start_date: str,
        end_date: str,
        min_rows: int,
        is_delta: bool = False,
    ) -> pd.DataFrame:

        ticker = ticker.strip().upper()

        df = pd.DataFrame(values)

        if df.empty:
            raise RuntimeError(f"TwelveData returned empty dataset for '{ticker}'.")

        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        if "date" not in df.columns:
            raise RuntimeError(
                f"TwelveData response missing 'datetime' field for '{ticker}'."
            )

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

        if df.empty:
            raise RuntimeError(
                f"TwelveData returned no data for '{ticker}' "
                f"in range [{start_date}, {end_date}]."
            )

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"TwelveData schema violation for '{ticker}': missing {missing}."
            )

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if df.empty:
            raise RuntimeError(f"All rows invalid after coercion for '{ticker}'.")

        df = df[df["high"] >= df["low"]]
        df = df[df["close"] > 0]

        if df.empty:
            raise RuntimeError(f"No valid OHLC bars remaining for '{ticker}'.")

        jumps = df["close"].pct_change().abs()
        if jumps.dropna().max() > 0.90:
            logger.warning("TwelveData: extreme price move detected for %s", ticker)

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        # FIX: Skip min_rows check for delta fetches — same logic as yahoo_provider.
        # Delta syncs return 4-5 rows legitimately. The old check always raised here.
        if is_delta:
            if len(df) == 0:
                raise RuntimeError(
                    f"TwelveData delta fetch returned zero rows for '{ticker}'."
                )
        else:
            if len(df) < min_rows:
                raise RuntimeError(
                    f"TwelveData insufficient history for '{ticker}': "
                    f"got {len(df)}, need {min_rows}."
                )

        df["ticker"] = ticker

        return df

    ############################################################
    # FETCH
    ############################################################

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
        **kwargs,
    ) -> pd.DataFrame:

        if interval not in self.INTERVAL_MAP:
            raise ValueError(
                f"TwelveData unsupported interval '{interval}'. "
                f"Allowed: {sorted(self.INTERVAL_MAP.keys())}"
            )

        td_interval = self.INTERVAL_MAP[interval]
        min_rows: int = kwargs.get("min_rows", self.DEFAULT_MIN_ROWS)

        # FIX: Detect delta fetch to skip min_rows check in _normalize
        is_delta = self._is_delta_fetch(start_date, end_date)

        params = {
            "symbol": ticker,
            "interval": td_interval,
            "start_date": start_date,
            "end_date": end_date,
            "outputsize": self.MAX_OUTPUTSIZE,
            "apikey": self.api_key,
            "format": "JSON",
        }

        data = self._call_api(params)

        if "values" not in data:
            raise RuntimeError(
                f"TwelveData response missing 'values' key for '{ticker}'. "
                f"Keys present: {list(data.keys())}"
            )

        df = self._normalize(
            values=data["values"],
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            min_rows=min_rows,
            is_delta=is_delta,
        )

        logger.info(
            "TwelveData served | ticker=%s interval=%s rows=%d range=[%s, %s]",
            ticker,
            interval,
            len(df),
            start_date,
            end_date,
        )

        return self.validate_contract(df)