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
    MAX_ROWS = 20000
    MAX_DAILY_MOVE = 0.85
    MAX_PRICE_JUMP = 5.0

    MIN_TRADING_DENSITY = 0.65
    MIN_UNIQUE_DAYS = 40

    MAX_WORKERS = int(os.getenv("MARKET_MAX_WORKERS", 4))
    MEMORY_CACHE_TTL = 30
    ENABLE_DISK_CACHE = True

    _PROVIDER = None
    _memory_cache = {}
    _cache_lock = Lock()

    ########################################################

    def __init__(self):

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        if MarketDataService._PROVIDER is None:
            MarketDataService._PROVIDER = MarketProviderRouter()

        self._fetcher = MarketDataService._PROVIDER

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
    # DATASET HASH
    ########################################################

    def _dataset_hash(self, df: pd.DataFrame):

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        payload = df[["date", "close", "volume"]].to_csv(index=False).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    ########################################################
    # DISK CACHE
    ########################################################

    def _disk_cache_path(self, ticker, start, end):
        fname = f"{ticker}_{start}_{end}.parquet"
        return self.DATA_DIR / fname

    ########################################################
    # VALIDATION
    ########################################################

    def _validate_dataset(self, df: pd.DataFrame, ticker: str, min_rows: int):

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("Market data empty or invalid structure.")

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

        # Remove forward leakage
        safe_cutoff = self._cap_to_safe_date(df["date"].max())
        df = df[df["date"] <= safe_cutoff]

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        if not np.isfinite(df[numeric_cols].values.astype(float)).all():
            raise RuntimeError("Non-finite prices detected.")

        # Remove zero-volume rows
        df = df[df["volume"] > 0]

        if df["close"].nunique() < 5:
            raise RuntimeError("Price series too flat.")

        # OHLC consistency
        if not ((df["high"] >= df["low"]) & 
                (df["high"] >= df["close"]) &
                (df["low"] <= df["close"])).all():
            raise RuntimeError("OHLC inconsistency detected.")

        # Clip extreme moves (split-like)
        pct = df["close"].pct_change().abs().fillna(0)

        if (pct > self.MAX_DAILY_MOVE).any():
            logger.warning("Extreme move detected in %s — smoothing.", ticker)
            df.loc[pct > self.MAX_DAILY_MOVE, "close"] = np.nan
            df["close"] = df["close"].ffill().bfill()

        # Large absolute price jumps
        if df["close"].max() / max(df["close"].min(), 1e-9) > self.MAX_PRICE_JUMP:
            logger.warning("Large price scale change detected in %s", ticker)

        # Density check
        if len(df) > 30:
            total_days = (df["date"].max() - df["date"].min()).days
            expected = total_days / 7 * 5
            density = len(df) / max(expected, 1)

            if density < self.MIN_TRADING_DENSITY:
                raise RuntimeError(
                    f"Low trading density ({density:.2f})"
                )

        if df["date"].nunique() < self.MIN_UNIQUE_DAYS:
            raise RuntimeError("Too few unique trading days.")

        if len(df) > self.MAX_ROWS:
            df = df.tail(self.MAX_ROWS)

        if len(df) < min_rows:
            raise RuntimeError(
                f"Insufficient history ({len(df)} < {min_rows})"
            )

        df["__dataset_hash"] = self._dataset_hash(df)

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
        retries=2
    ):

        last_error = None

        for attempt in range(retries):

            try:

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

                return validated

            except Exception as e:

                last_error = e

                sleep = 0.4 + random.uniform(0.1, 0.4)

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

        disk_path = self._disk_cache_path(
            ticker,
            start_date,
            end_date_str
        )

        if self.ENABLE_DISK_CACHE and disk_path.exists():
            try:
                df = pd.read_parquet(disk_path)
                return df
            except Exception:
                pass

        logger.info("Fetching dataset for %s", ticker)

        df = self._fetch_with_retry(
            ticker,
            start_date,
            end_date_str,
            interval,
            min_history
        )

        if self.ENABLE_DISK_CACHE:
            try:
                df.to_parquet(disk_path)
            except Exception:
                pass

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
            max(2, min(len(tickers), 6))
        )

        logger.info(
            "Batch fetch | tickers=%d | workers=%d",
            len(tickers),
            worker_cap
        )

        with ThreadPoolExecutor(max_workers=worker_cap) as executor:

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
                    failures[ticker] = str(e)
                    logger.error(
                        "Batch fetch failed: %s | %s",
                        ticker,
                        str(e)
                    )

        if not results:
            raise RuntimeError("All tickers failed during batch fetch.")

        if failures:
            logger.warning(
                "Batch partial failure | success=%d | failed=%d",
                len(results),
                len(failures)
            )

        return results