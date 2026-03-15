"""
MarketSentinel v4.1.0

Service layer between MarketProviderRouter and the rest of the application.

Responsibilities:
    - Ticker sanitization
    - Safe end-date capping (SAFE_LAG_DAYS lag)
    - In-memory LRU cache with TTL
    - Global rate limiting across batch calls
    - Light dataset validation (schema + density — no data repair)
    - Parallel batch fetching via ThreadPoolExecutor

Data repair (OHLC fix, extreme moves, NaN fill) is intentionally NOT done here.
It is already handled upstream in data_fetcher → yahoo_provider → router.
"""

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
    """
    Primary entry point for all market data requests in the application.

    Uses a singleton MarketProviderRouter (class-level) to avoid creating
    multiple router instances with separate provider state.
    """

    DATA_DIR = Path("data/lake")

    # ── Schema ───────────────────────────────────────────────────────────────
    REQUIRED_COLUMNS = {
        "ticker", "date", "open", "high", "low", "close", "volume"
    }

    # ── Data quality thresholds ───────────────────────────────────────────────
    DEFAULT_MIN_HISTORY_ROWS = 60
    SAFE_LAG_DAYS            = 2       # days lag before today for safe end_date
    MAX_ROWS                 = 20_000
    MIN_TRADING_DENSITY      = 0.55    # warn (not fail) below this density
    MIN_UNIQUE_DAYS          = 40

    # ── Concurrency ──────────────────────────────────────────────────────────
    MAX_WORKERS = int(os.getenv("MARKET_MAX_WORKERS", "4"))

    # ── Memory cache ─────────────────────────────────────────────────────────
    MEMORY_CACHE_TTL = 30    # seconds
    MEMORY_CACHE_MAX = 200   # max items before LRU eviction

    # ── Rate limiting ─────────────────────────────────────────────────────────
    GLOBAL_RATE_LIMIT_PER_SEC = 3
    BATCH_SPACING_SECONDS     = 0.15
    MAX_RETRY_BACKOFF         = 2.5

    # ── Soft-fail mode ────────────────────────────────────────────────────────
    SOFT_FAIL_MODE = os.getenv("MARKET_SOFT_FAIL", "1") == "1"

    # ── Class-level singletons (shared across instances) ─────────────────────
    _PROVIDER:       Optional[MarketProviderRouter] = None
    _memory_cache:   OrderedDict                    = OrderedDict()
    _cache_lock:     Lock                           = Lock()
    _rate_lock:      Lock                           = Lock()
    _recent_requests: deque                         = deque()

    # ────────────────────────────────────────────────────────────────────────
    # INIT
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Singleton router — one instance shared across all service objects
        if MarketDataService._PROVIDER is None:
            MarketDataService._PROVIDER = MarketProviderRouter()

        self._fetcher = MarketDataService._PROVIDER

    # ────────────────────────────────────────────────────────────────────────
    # RATE LIMITING
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def _respect_rate_limit(cls) -> None:
        """
        Enforce GLOBAL_RATE_LIMIT_PER_SEC across all concurrent workers.
        Uses a sliding 1-second window.
        """
        with cls._rate_lock:
            now = time.time()

            # Evict timestamps older than 1 second
            while cls._recent_requests and now - cls._recent_requests[0] > 1.0:
                cls._recent_requests.popleft()

            if len(cls._recent_requests) >= cls.GLOBAL_RATE_LIMIT_PER_SEC:
                sleep_for = 1.0 - (now - cls._recent_requests[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)

            cls._recent_requests.append(time.time())

    # ────────────────────────────────────────────────────────────────────────
    # TICKER SANITIZATION
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_ticker(ticker: str) -> str:
        """
        Validate ticker format. Raises RuntimeError on unsafe input.
        Accepts: A-Z, 0-9, '.', '_', '-' — max 12 characters.
        """
        ticker = str(ticker).upper().strip()
        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):
            raise RuntimeError(
                f"Unsafe or invalid ticker format: '{ticker}'"
            )
        return ticker

    # ────────────────────────────────────────────────────────────────────────
    # SAFE DATE CAP
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def _cap_to_safe_date(cls, date_input: str) -> pd.Timestamp:
        """
        Cap end_date to (today - SAFE_LAG_DAYS) to avoid requesting
        data that may not yet be available from providers.
        """
        requested    = pd.Timestamp(date_input).normalize()
        safe_cutoff  = (
            pd.Timestamp.now(tz="UTC")
            .tz_localize(None)
            .normalize()
            - pd.Timedelta(days=cls.SAFE_LAG_DAYS)
        )
        return min(requested, safe_cutoff)

    # ────────────────────────────────────────────────────────────────────────
    # DATASET HASH  (fingerprint for debugging)
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dataset_hash(df: pd.DataFrame) -> str:
        df_sorted = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        payload   = df_sorted[["date", "close", "volume"]].to_csv(index=False).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    # ────────────────────────────────────────────────────────────────────────
    # DATASET VALIDATION  (schema + density — no data repair)
    # ────────────────────────────────────────────────────────────────────────

    def _validate_dataset(
        self,
        df:       pd.DataFrame,
        ticker:   str,
        min_rows: int,
    ) -> pd.DataFrame:
        """
        Validate schema, column types, trading density, and row count.

        Intentionally does NOT repair OHLC or extreme moves —
        that is already handled upstream in:
            data_fetcher → yahoo_provider → MarketProviderRouter
        """

        if df is None or df.empty:
            raise RuntimeError(f"Market data empty for {ticker}.")

        df = df.copy()

        # ── Schema ────────────────────────────────────────────────────────────
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Schema violation for {ticker}. Missing: {missing}"
            )

        df["ticker"] = ticker

        # ── Date normalisation (keep UTC-aware) ───────────────────────────────
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").drop_duplicates("date")

        # ── Numeric coercion ──────────────────────────────────────────────────
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=numeric_cols)

        # ── Finite guard ──────────────────────────────────────────────────────
        if not np.isfinite(df[numeric_cols].values).all():
            raise RuntimeError(f"Non-finite values detected for {ticker}.")

        # ── Flat price guard ──────────────────────────────────────────────────
        if df["close"].nunique() < 5:
            raise RuntimeError(
                f"Flat price series detected for {ticker} — possible bad data."
            )

        # ── Negative volume guard (allow zero — some ETFs / holidays) ─────────
        df = df[df["volume"] >= 0]

        # ── Cap to MAX_ROWS ───────────────────────────────────────────────────
        if len(df) > self.MAX_ROWS:
            df = df.tail(self.MAX_ROWS).reset_index(drop=True)

        # ── Trading density warning (informational only) ──────────────────────
        span_days = (df["date"].max() - df["date"].min()).days + 1
        if span_days > 0:
            density = df["date"].nunique() / span_days
            if density < self.MIN_TRADING_DENSITY:
                logger.warning(
                    "Low trading density for %s: %.2f (threshold=%.2f)",
                    ticker, density, self.MIN_TRADING_DENSITY,
                )

        # ── Minimum rows gate ─────────────────────────────────────────────────
        if len(df) < min_rows:
            soft_threshold = int(min_rows * 0.7)
            if self.SOFT_FAIL_MODE and len(df) >= soft_threshold:
                logger.warning(
                    "Short history for %s accepted in soft-fail mode "
                    "(%d rows, need %d).",
                    ticker, len(df), min_rows,
                )
            else:
                raise RuntimeError(
                    f"Insufficient history for {ticker}: "
                    f"got {len(df)}, need {min_rows}."
                )

        df["__dataset_hash"] = self._dataset_hash(df)

        logger.debug(
            "Dataset validated | ticker=%s rows=%d hash=%s",
            ticker, len(df), df["__dataset_hash"].iloc[0],
        )

        return df.reset_index(drop=True)

    # ────────────────────────────────────────────────────────────────────────
    # MEMORY CACHE
    # ────────────────────────────────────────────────────────────────────────

    def _cache_key(
        self,
        ticker:      str,
        start:       str,
        end:         str,
        interval:    str,
        min_history: int,
    ) -> str:
        return f"{ticker}_{start}_{end}_{interval}_{min_history}"

    def _get_from_memory_cache(self, key: str) -> Optional[pd.DataFrame]:
        with self._cache_lock:
            item = self._memory_cache.get(key)
            if not item:
                return None
            df, ts = item
            if time.time() - ts > self.MEMORY_CACHE_TTL:
                del self._memory_cache[key]
                return None
            # Move to end (most recently used)
            self._memory_cache.move_to_end(key)
            return df.copy()

    def _set_memory_cache(self, key: str, df: pd.DataFrame) -> None:
        with self._cache_lock:
            # LRU eviction — remove least recently used
            if len(self._memory_cache) >= self.MEMORY_CACHE_MAX:
                self._memory_cache.popitem(last=False)
            self._memory_cache[key] = (df.copy(), time.time())

    # ────────────────────────────────────────────────────────────────────────
    # SINGLE TICKER FETCH
    # ────────────────────────────────────────────────────────────────────────

    def get_price_data(
        self,
        ticker:      str,
        start_date:  str,
        end_date:    str,
        interval:    str             = "1d",
        min_history: Optional[int]   = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Parameters
        ----------
        ticker      : Stock symbol, e.g. "AAPL"
        start_date  : ISO string, e.g. "2023-01-01"
        end_date    : ISO string — capped to (today - SAFE_LAG_DAYS)
        interval    : e.g. "1d", "1h", "15m"
        min_history : Minimum acceptable bar count
        """
        ticker = self._sanitize_ticker(ticker)

        safe_end    = self._cap_to_safe_date(end_date)
        end_str     = safe_end.strftime("%Y-%m-%d")
        start_str   = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        if min_history is None:
            min_history = self.DEFAULT_MIN_HISTORY_ROWS

        cache_key = self._cache_key(ticker, start_str, end_str, interval, min_history)

        cached = self._get_from_memory_cache(cache_key)
        if cached is not None:
            logger.debug("Memory cache hit | ticker=%s", ticker)
            return cached

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
            "Market data served | ticker=%s interval=%s rows=%d",
            ticker, interval, len(df),
        )

        return df

    # ────────────────────────────────────────────────────────────────────────
    # BATCH FETCH  (parallel)
    # ────────────────────────────────────────────────────────────────────────

    def get_price_data_batch(
        self,
        tickers:     List[str],
        start_date:  str,
        end_date:    str,
        interval:    str           = "1d",
        min_history: Optional[int] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Fetch OHLCV data for multiple tickers in parallel.

        Returns
        -------
        results  : {ticker: DataFrame} for successful fetches
        failures : {ticker: error_message} for failed fetches
        Raises RuntimeError if ALL tickers fail.
        """
        results:  Dict[str, pd.DataFrame] = {}
        failures: Dict[str, str]          = {}

        # Deduplicate tickers while preserving order
        tickers = list(dict.fromkeys(tickers))

        worker_cap = min(self.MAX_WORKERS, max(2, min(len(tickers), 5)))

        with ThreadPoolExecutor(max_workers=worker_cap) as executor:
            futures: Dict = {}

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
                # Small spacing to avoid hammering the rate limiter at once
                time.sleep(self.BATCH_SPACING_SECONDS)

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker] = future.result()
                except Exception as exc:
                    failures[ticker] = str(exc)
                    logger.warning(
                        "Batch fetch failed | ticker=%s | error=%s",
                        ticker, exc,
                    )

        if not results:
            raise RuntimeError(
                f"All {len(tickers)} tickers failed during batch fetch. "
                f"Errors: {failures}"
            )

        if failures:
            logger.warning(
                "Batch partial failure | success=%d | failed=%d | tickers=%s",
                len(results), len(failures), list(failures.keys()),
            )

        return results, failures