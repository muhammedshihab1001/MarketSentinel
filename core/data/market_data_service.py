import logging
from pathlib import Path
import pandas as pd
import time
import numpy as np
import hashlib
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
from collections import deque, OrderedDict

from core.data.providers.market.router import MarketProviderRouter

logger = logging.getLogger(__name__)


class MarketDataService:

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

    DEFAULT_MIN_HISTORY_ROWS = 60
    SAFE_LAG_DAYS = 2
    MAX_ROWS = 20000
    MAX_DAILY_MOVE = 0.85

    MIN_TRADING_DENSITY = 0.55
    MIN_UNIQUE_DAYS = 40

    MAX_WORKERS = int(os.getenv("MARKET_MAX_WORKERS", 4))
    MEMORY_CACHE_TTL = 30
    MEMORY_CACHE_MAX = 200

    ENABLE_DISK_CACHE = True
    SOFT_FAIL_MODE = os.getenv("MARKET_SOFT_FAIL", "1") == "1"

    GLOBAL_RATE_LIMIT_PER_SEC = 3
    BATCH_SPACING_SECONDS = 0.15
    MAX_RETRY_BACKOFF = 2.5
    PROVIDER_COOLDOWN_SECONDS = 5

    _PROVIDER = None
    _memory_cache = OrderedDict()

    _cache_lock = Lock()
    _rate_lock = Lock()

    _recent_requests = deque()
    _provider_cooldown_until = 0

    ########################################################

    def __init__(self):

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        if MarketDataService._PROVIDER is None:
            MarketDataService._PROVIDER = MarketProviderRouter()

        self._fetcher = MarketDataService._PROVIDER

    ########################################################
    # RATE LIMITING
    ########################################################

    @classmethod
    def _respect_rate_limit(cls):

        with cls._rate_lock:

            now = time.time()

            if now < cls._provider_cooldown_until:
                time.sleep(cls._provider_cooldown_until - now)

            while cls._recent_requests and now - cls._recent_requests[0] > 1:
                cls._recent_requests.popleft()

            if len(cls._recent_requests) >= cls.GLOBAL_RATE_LIMIT_PER_SEC:

                sleep = 1 - (now - cls._recent_requests[0])

                if sleep > 0:
                    time.sleep(sleep)

            cls._recent_requests.append(time.time())

    ########################################################
    # SANITIZATION
    ########################################################

    @staticmethod
    def _sanitize_ticker(ticker: str):

        ticker = str(ticker).upper().strip()

        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):
            raise RuntimeError(f"Unsafe ticker: {ticker}")

        return ticker

    ########################################################
    # SAFE DATE CAP
    ########################################################

    @classmethod
    def _cap_to_safe_date(cls, date_input):

        requested = pd.Timestamp(date_input).tz_localize(None)

        safe_cutoff = (
            pd.Timestamp.utcnow()
            .tz_localize(None)
            .normalize()
            - pd.Timedelta(days=cls.SAFE_LAG_DAYS)
        )

        return min(requested, safe_cutoff)

    ########################################################
    # DATASET HASH
    ########################################################

    def _dataset_hash(self, df: pd.DataFrame):

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        payload = df[["date", "close", "volume"]].to_csv(index=False).encode()

        return hashlib.sha256(payload).hexdigest()[:16]

    ########################################################
    # VALIDATION
    ########################################################

    def _validate_dataset(self, df: pd.DataFrame, ticker: str, min_rows: int):

        if df is None or df.empty:
            raise RuntimeError("Market data empty.")

        df = df.copy()

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(f"Schema violation. Missing={missing}")

        df["ticker"] = ticker

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        df = df.dropna(subset=["date"])
        df = df.sort_values("date").drop_duplicates("date")

        df = df[df["volume"] > 0]

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        if not np.isfinite(df[numeric_cols].values).all():
            raise RuntimeError("Non-finite values detected.")

        if df["close"].nunique() < 5:
            raise RuntimeError("Flat price series.")

        unique_days = df["date"].nunique()

        span_days = (df["date"].max() - df["date"].min()).days + 1

        if span_days > 0:

            density = unique_days / span_days

            if density < self.MIN_TRADING_DENSITY:
                logger.warning("Low trading density detected for %s", ticker)

        pct = df["close"].pct_change().abs().fillna(0)

        if (pct > self.MAX_DAILY_MOVE).any():

            df.loc[pct > self.MAX_DAILY_MOVE, ["open", "high", "low", "close"]] = np.nan

            df[["open", "high", "low", "close"]] = df[
                ["open", "high", "low", "close"]
            ].ffill().bfill()

        if len(df) < min_rows:

            if self.SOFT_FAIL_MODE and len(df) >= int(min_rows * 0.7):

                logger.warning(
                    "History slightly short for %s but accepted in soft mode.",
                    ticker
                )

            else:
                raise RuntimeError("Insufficient history.")

        df["__dataset_hash"] = self._dataset_hash(df)

        return df.reset_index(drop=True)

    ########################################################
    # MEMORY CACHE
    ########################################################

    def _cache_key(self, ticker, start, end, interval, min_history):
        return f"{ticker}_{start}_{end}_{interval}_{min_history}"

    def _get_from_memory_cache(self, key):

        with self._cache_lock:

            item = self._memory_cache.get(key)

            if not item:
                return None

            df, ts = item

            if time.time() - ts > self.MEMORY_CACHE_TTL:

                del self._memory_cache[key]

                return None

            return df.copy()

    def _set_memory_cache(self, key, df):

        with self._cache_lock:

            if len(self._memory_cache) >= self.MEMORY_CACHE_MAX:
                self._memory_cache.popitem(last=False)

            self._memory_cache[key] = (df.copy(), time.time())

    ########################################################
    # SINGLE FETCH
    ########################################################

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        min_history: int | None = None
    ):

        ticker = self._sanitize_ticker(ticker)

        end_date = self._cap_to_safe_date(end_date)

        if min_history is None:
            min_history = self.DEFAULT_MIN_HISTORY_ROWS

        start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        end_date_str = end_date.strftime("%Y-%m-%d")

        cache_key = self._cache_key(
            ticker,
            start_date,
            end_date_str,
            interval,
            min_history
        )

        cached = self._get_from_memory_cache(cache_key)

        if cached is not None:
            return cached

        df = self._fetcher.fetch(
            ticker,
            start_date,
            end_date_str,
            interval,
            min_rows=min_history
        )

        df = self._validate_dataset(df, ticker, min_history)

        self._set_memory_cache(cache_key, df)

        return df

    ########################################################
    # PARALLEL BATCH
    ########################################################

    def get_price_data_batch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        min_history: int | None = None
    ):

        results = {}
        failures = {}

        tickers = list(dict.fromkeys(tickers))

        worker_cap = min(
            self.MAX_WORKERS,
            max(2, min(len(tickers), 5))
        )

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
                        min_history
                    )
                ] = ticker

                time.sleep(self.BATCH_SPACING_SECONDS)

            for future in as_completed(futures):

                ticker = futures[future]

                try:

                    results[ticker] = future.result()

                except Exception as e:

                    failures[ticker] = str(e)

        if not results:
            raise RuntimeError("All tickers failed during batch fetch.")

        if failures:

            logger.warning(
                "Batch partial failure | success=%d | failed=%d",
                len(results),
                len(failures)
            )

        return results, failures