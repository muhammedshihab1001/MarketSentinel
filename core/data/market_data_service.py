import hashlib
import logging
import os
import re
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.data.providers.market.router import MarketProviderRouter

logger = logging.getLogger(__name__)


class MarketDataService:

    DATA_DIR = Path("data/lake")

    REQUIRED_COLUMNS = {
        "ticker", "date", "open", "high", "low", "close", "volume"
    }

    DEFAULT_MIN_HISTORY_ROWS = 60
    SAFE_LAG_DAYS = 2
    MAX_ROWS = 20_000
    MIN_TRADING_DENSITY = 0.55
    MIN_UNIQUE_DAYS = 40

    MAX_WORKERS = int(os.getenv("MARKET_MAX_WORKERS", "4"))

    MEMORY_CACHE_TTL = 30
    MEMORY_CACHE_MAX = 200

    GLOBAL_RATE_LIMIT_PER_SEC = 3
    BATCH_SPACING_SECONDS = 0.15
    MAX_RETRY_BACKOFF = 2.5

    SOFT_FAIL_MODE = os.getenv("MARKET_SOFT_FAIL", "1") == "1"

    _PROVIDER: Optional[MarketProviderRouter] = None

    _memory_cache: OrderedDict = OrderedDict()
    _cache_lock: Lock = Lock()

    _rate_lock: Lock = Lock()
    _recent_requests: deque = deque()

    # NEW: prevent duplicate concurrent fetch
    _inflight_requests: Dict[str, bool] = {}
    _inflight_lock: Lock = Lock()

    # NEW: cache stats (debugging)
    _cache_hits = 0
    _cache_misses = 0

    def __init__(self) -> None:

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        if MarketDataService._PROVIDER is None:
            MarketDataService._PROVIDER = MarketProviderRouter()

        self._fetcher = MarketDataService._PROVIDER

    # --------------------------------------------------
    # RATE LIMIT
    # --------------------------------------------------

    @classmethod
    def _respect_rate_limit(cls):

        with cls._rate_lock:

            now = time.time()

            while cls._recent_requests and now - cls._recent_requests[0] > 1.0:
                cls._recent_requests.popleft()

            if len(cls._recent_requests) >= cls.GLOBAL_RATE_LIMIT_PER_SEC:

                sleep_for = 1.0 - (now - cls._recent_requests[0])

                if sleep_for > 0:
                    time.sleep(sleep_for)

            cls._recent_requests.append(time.time())

    # --------------------------------------------------
    # TICKER VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _sanitize_ticker(ticker: str) -> str:

        ticker = str(ticker).upper().strip()

        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):

            raise RuntimeError(
                f"Unsafe or invalid ticker format: '{ticker}'"
            )

        return ticker

    # --------------------------------------------------
    # SAFE DATE
    # --------------------------------------------------

    @classmethod
    def _cap_to_safe_date(cls, date_input: str) -> pd.Timestamp:

        requested = pd.Timestamp(date_input).normalize()

        safe_cutoff = (
            pd.Timestamp.now(tz="UTC")
            .tz_localize(None)
            .normalize()
            - pd.Timedelta(days=cls.SAFE_LAG_DAYS)
        )

        return min(requested, safe_cutoff)

    # --------------------------------------------------
    # DATASET HASH
    # --------------------------------------------------

    @staticmethod
    def _dataset_hash(df: pd.DataFrame) -> str:

        df_sorted = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        payload = df_sorted[["date", "close", "volume"]].to_csv(
            index=False
        ).encode()

        return hashlib.sha256(payload).hexdigest()[:16]

    # --------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------

    def _validate_dataset(self, df: pd.DataFrame, ticker: str, min_rows: int):

        if df is None or df.empty:
            raise RuntimeError(f"Market data empty for {ticker}")

        df = df.copy()

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Schema violation for {ticker}. Missing: {missing}"
            )

        df["ticker"] = ticker

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

        df = df.dropna(subset=["date"])

        df = df.sort_values("date").drop_duplicates("date")

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        if not np.isfinite(df[numeric_cols].values).all():
            raise RuntimeError(f"Non-finite values detected for {ticker}")

        if df["close"].nunique() < 5:
            raise RuntimeError(
                f"Flat price series detected for {ticker}"
            )

        df = df[df["volume"] >= 0]

        if len(df) > self.MAX_ROWS:
            df = df.tail(self.MAX_ROWS).reset_index(drop=True)

        span_days = (df["date"].max() - df["date"].min()).days + 1

        if span_days > 0:

            density = df["date"].nunique() / span_days

            if density < self.MIN_TRADING_DENSITY:

                logger.warning(
                    "Low trading density for %s: %.2f",
                    ticker,
                    density,
                )

        if len(df) < min_rows:

            soft_threshold = int(min_rows * 0.7)

            if self.SOFT_FAIL_MODE and len(df) >= soft_threshold:

                logger.warning(
                    "Short history accepted for %s (%d rows)",
                    ticker,
                    len(df),
                )

            else:

                raise RuntimeError(
                    f"Insufficient history for {ticker}: got {len(df)}"
                )

        df["__dataset_hash"] = self._dataset_hash(df)

        return df.reset_index(drop=True)

    # --------------------------------------------------
    # CACHE
    # --------------------------------------------------

    def _cache_key(self, ticker, start, end, interval, min_history):

        return f"{ticker}_{start}_{end}_{interval}_{min_history}"

    def _get_from_memory_cache(self, key):

        with self._cache_lock:

            item = self._memory_cache.get(key)

            if not item:
                MarketDataService._cache_misses += 1
                return None

            df, ts = item

            if time.time() - ts > self.MEMORY_CACHE_TTL:

                del self._memory_cache[key]

                MarketDataService._cache_misses += 1

                return None

            self._memory_cache.move_to_end(key)

            MarketDataService._cache_hits += 1

            return df.copy()

    def _set_memory_cache(self, key, df):

        with self._cache_lock:

            if len(self._memory_cache) >= self.MEMORY_CACHE_MAX:
                self._memory_cache.popitem(last=False)

            self._memory_cache[key] = (df.copy(), time.time())

    # --------------------------------------------------
    # SINGLE FETCH
    # --------------------------------------------------

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        min_history: Optional[int] = None,
    ):

        ticker = self._sanitize_ticker(ticker)

        safe_end = self._cap_to_safe_date(end_date)

        end_str = safe_end.strftime("%Y-%m-%d")

        start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        if min_history is None:
            min_history = self.DEFAULT_MIN_HISTORY_ROWS

        cache_key = self._cache_key(
            ticker,
            start_str,
            end_str,
            interval,
            min_history,
        )

        cached = self._get_from_memory_cache(cache_key)

        if cached is not None:
            return cached

        with self._inflight_lock:

            if cache_key in self._inflight_requests:

                while cache_key in self._inflight_requests:
                    time.sleep(0.05)

                cached = self._get_from_memory_cache(cache_key)

                if cached is not None:
                    return cached

            self._inflight_requests[cache_key] = True

        try:

            self._respect_rate_limit()

            df = self._fetcher.fetch(
                ticker,
                start_str,
                end_str,
                interval,
                min_rows=min_history,
            )

            df = self._validate_dataset(df, ticker, min_history)

            self._set_memory_cache(cache_key, df)

            logger.info(
                "Market data served | ticker=%s rows=%d",
                ticker,
                len(df),
            )

            return df

        finally:

            with self._inflight_lock:

                self._inflight_requests.pop(cache_key, None)

    # --------------------------------------------------
    # BATCH FETCH
    # --------------------------------------------------

    def get_price_data_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        min_history: Optional[int] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:

        results = {}
        failures = {}

        tickers = list(dict.fromkeys(tickers))

        worker_cap = min(self.MAX_WORKERS, max(2, min(len(tickers), 5)))

        with ThreadPoolExecutor(max_workers=worker_cap) as executor:

            futures = {}

            for ticker in tickers:

                futures[
                    executor.submit(
                        self.get_price_data,
                        ticker,
                        start_date,
                        end_date,
                        interval,
                        min_history,
                    )
                ] = ticker

                time.sleep(self.BATCH_SPACING_SECONDS)

            for future in as_completed(futures):

                ticker = futures[future]

                try:

                    results[ticker] = future.result()

                except Exception as exc:

                    failures[ticker] = str(exc)

                    logger.warning(
                        "Batch fetch failed | ticker=%s | error=%s",
                        ticker,
                        exc,
                    )

        if not results:

            raise RuntimeError(
                f"All {len(tickers)} tickers failed during batch fetch."
            )

        if failures:

            logger.warning(
                "Batch partial failure | success=%d failed=%d",
                len(results),
                len(failures),
            )

        return results, failures