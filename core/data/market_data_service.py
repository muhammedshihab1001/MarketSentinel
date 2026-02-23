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
    MAX_ROWS = 15000
    MIN_FILE_BYTES = 5_000
    MAX_DAILY_MOVE = 0.85

    MAX_WORKERS = int(os.getenv("MARKET_MAX_WORKERS", 6))
    MEMORY_CACHE_TTL = 30

    FAST_PROVIDER_MODE = os.getenv("MARKET_FAST_MODE", "true").lower() == "true"

    _PROVIDER = None
    _memory_cache = {}
    _cache_lock = Lock()

    ########################################################

    def __init__(self):

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        if MarketDataService._PROVIDER is None:
            MarketDataService._PROVIDER = MarketProviderRouter()

        self._fetcher = MarketDataService._PROVIDER

        if self.FAST_PROVIDER_MODE and len(self._fetcher.providers) == 1:
            self.MAX_WORKERS = min(self.MAX_WORKERS, 2)
            logger.info(
                "Single-provider FAST mode → worker cap set to %s",
                self.MAX_WORKERS
            )

        self.SCHEMA_HASH = hashlib.sha256(
            ",".join(sorted(self.REQUIRED_COLUMNS)).encode()
        ).hexdigest()[:10]

    ########################################################

    @staticmethod
    def _sanitize_ticker(ticker: str):

        ticker = ticker.upper()

        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):
            raise RuntimeError(f"Unsafe ticker: {ticker}")

        return ticker

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

    def _dataset_hash(self, df: pd.DataFrame):

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        payload = df.to_csv(index=False).encode()

        return hashlib.sha256(payload).hexdigest()[:16]

    ########################################################
    # 🔥 HARDENED VALIDATION
    ########################################################

    def _validate_dataset(self, df: pd.DataFrame, ticker: str, min_rows: int):

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("Market data empty or invalid structure.")

        df = df.copy()

        # enforce required columns existence
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(f"Schema violation. Missing={missing}")

        df["ticker"] = ticker

        # normalize date safely
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        if df["date"].isna().any():
            raise RuntimeError("Invalid date values detected.")

        numeric_cols = ["open", "high", "low", "close", "volume"]

        # strict numeric enforcement
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[numeric_cols].isna().any().any():
            raise RuntimeError("NaN values detected in numeric columns.")

        # 🔥 safer finite check
        numeric_array = df[numeric_cols].values.astype(float)

        if not np.isfinite(numeric_array).all():
            raise RuntimeError("Non-finite prices detected.")

        df = df.sort_values("date").reset_index(drop=True)

        # daily jump guard
        pct = df["close"].pct_change().abs().fillna(0)

        if (pct > self.MAX_DAILY_MOVE).any():
            raise RuntimeError("Extreme price jump detected.")

        if len(df) > self.MAX_ROWS:
            df = df.tail(self.MAX_ROWS)

        if len(df) < min_rows:
            raise RuntimeError(
                f"Insufficient history ({len(df)} < {min_rows})"
            )

        return df.reset_index(drop=True)

    ########################################################
    # RETRY FETCH
    ########################################################

    def _fetch_with_retry(
        self,
        ticker,
        start,
        end,
        interval,
        min_history,
        retries=3
    ):

        last_error = None

        for attempt in range(retries):

            try:

                if self.FAST_PROVIDER_MODE:
                    df = self._fetcher.fetch(
                        ticker,
                        start,
                        end,
                        interval,
                        min_rows=min_history,
                        provider="yahoo"
                    )
                else:
                    df = self._fetcher.fetch(
                        ticker,
                        start,
                        end,
                        interval,
                        min_rows=min_history
                    )

                validated = self._validate_dataset(
                    df,
                    ticker,
                    min_history
                )

                validated["__dataset_hash"] = self._dataset_hash(validated)

                return validated

            except Exception as e:

                last_error = e

                sleep = (2 ** attempt) + random.uniform(0, 0.5)

                logger.warning(
                    "Fetch failed (%s) attempt %d/%d | %.2fs | %s",
                    ticker,
                    attempt + 1,
                    retries,
                    sleep,
                    str(e)
                )

                time.sleep(sleep)

        raise RuntimeError(
            f"Market fetch failed after retries: {ticker}"
        ) from last_error

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
            self._memory_cache[key] = (df.copy(), time.time())

    ########################################################
    # PUBLIC API
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

        cache_key = self._cache_key(
            ticker,
            start_date,
            end_date.strftime("%Y-%m-%d"),
            interval,
            min_history
        )

        cached = self._get_from_memory_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(
            "Fetching dataset for %s (min_history=%d)",
            ticker,
            min_history
        )

        df = self._fetch_with_retry(
            ticker,
            start_date,
            end_date.strftime("%Y-%m-%d"),
            interval,
            min_history
        )

        self._set_memory_cache(cache_key, df)

        return df

    ########################################################
    # PARALLEL BATCH FETCH
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

        tickers = list(dict.fromkeys(tickers))  # preserve order safely

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:

            futures = {
                executor.submit(
                    self.get_price_data,
                    ticker,
                    start_date,
                    end_date,
                    interval,
                    min_history
                ): ticker
                for ticker in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    logger.error(
                        "Batch fetch failed: %s | %s",
                        ticker,
                        str(e)
                    )
                    raise

        return {t: results[t] for t in tickers if t in results}