import yfinance as yf
import pandas as pd
import time
import random
import os
import hashlib
import logging

from datetime import datetime, timedelta

logger = logging.getLogger("marketsentinel.fetcher")


class StockPriceFetcher:
    """
    Institutional Market Data Fetcher.

    Guarantees:
    - timezone normalization (UTC naive)
    - numeric enforcement
    - deterministic datasets
    - cache safety
    - provider fallback
    """

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    MIN_ROWS = 120
    MAX_RETRIES = 5
    BASE_SLEEP = 1.5

    CACHE_DIR = "data/cache"
    DATASET_DIR = "data/datasets"

    # -----------------------------------------------------

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.DATASET_DIR, exist_ok=True)

    # =====================================================
    # NORMALIZATION (VERY IMPORTANT)
    # =====================================================

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.rename(columns=str.lower)

        if "adj close" in df.columns:
            df.drop(columns=["adj close"], inplace=True)

        df["date"] = pd.to_datetime(df["date"], utc=True)\
            .dt.tz_convert(None)

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        return df

    # =====================================================

    def _validate_dataset(self, df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Provider returned empty dataset.")

        df = self._normalize_schema(df)

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Provider schema violation: missing={missing}"
            )

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if (df["high"] < df["low"]).any():
            raise RuntimeError("High < Low detected.")

        df = df.drop_duplicates("date")

        df = df.sort_values("date")

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Non-monotonic timestamps detected.")

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Dataset too small for safe usage.")

        return df.reset_index(drop=True)

    # =====================================================

    def _cache_key(self, ticker, start, end, interval):

        raw = f"{ticker}_{start}_{end}_{interval}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # =====================================================

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        snapshot: bool = False
    ) -> pd.DataFrame:

        self._validate_dates(start_date, end_date)

        cache_name = self._cache_key(
            ticker,
            start_date,
            end_date,
            interval
        )

        cache_file = f"{self.CACHE_DIR}/{cache_name}.parquet"

        # ==========================================
        # LOAD CACHE
        # ==========================================

        if os.path.exists(cache_file):

            try:

                cached = pd.read_parquet(cache_file)
                cached = self._validate_dataset(cached)

                logger.info(f"Loaded dataset from cache → {ticker}")

                return cached

            except Exception:

                logger.exception("Cache corrupted — rebuilding.")

                try:
                    os.remove(cache_file)
                except Exception:
                    pass

        # ==========================================
        # FULL DOWNLOAD
        # ==========================================

        df = self._fetch_from_providers(
            ticker,
            start_date,
            end_date,
            interval
        )

        df = self._validate_dataset(df)

        df.to_parquet(cache_file, index=False)

        logger.info(f"Saved cache for {ticker}")

        if snapshot:
            self._snapshot_dataset(ticker, df)

        return df

    # -----------------------------------------------------

    def _fetch_from_providers(
        self,
        ticker,
        start_date,
        end_date,
        interval
    ):

        providers = [
            self._fetch_yahoo,
            self._fetch_stooq
        ]

        last_exception = None

        for provider in providers:

            try:
                return provider(
                    ticker,
                    start_date,
                    end_date,
                    interval
                )

            except Exception as e:

                last_exception = e

                logger.warning(
                    f"Provider failed: {provider.__name__}"
                )

        raise RuntimeError(
            f"All providers failed for {ticker}. "
            f"Last error: {last_exception}"
        )

    # -----------------------------------------------------

    def _fetch_yahoo(
        self,
        ticker,
        start_date,
        end_date,
        interval
    ):

        for attempt in range(self.MAX_RETRIES):

            try:

                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False
                )

                if df.empty:
                    raise RuntimeError("Yahoo returned empty dataframe")

                df.reset_index(inplace=True)

                return df

            except Exception:

                sleep_time = (
                    self.BASE_SLEEP * (2 ** attempt)
                    + random.uniform(0, 1)
                )

                logger.warning(
                    f"[Yahoo] retry {attempt+1}/{self.MAX_RETRIES} "
                    f"in {round(sleep_time,2)}s"
                )

                time.sleep(sleep_time)

        raise RuntimeError("Yahoo failed")

    # -----------------------------------------------------

    def _fetch_stooq(
        self,
        ticker,
        start_date,
        end_date,
        interval
    ):

        tk = ticker.lower()

        url = f"https://stooq.com/q/d/l/?s={tk}&i=d"

        df = pd.read_csv(url)

        if df.empty:
            raise RuntimeError("Stooq returned empty dataframe")

        df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        df = df.loc[
            (pd.to_datetime(df["date"]) >= start) &
            (pd.to_datetime(df["date"]) <= end)
        ]

        return df

    # -----------------------------------------------------

    @staticmethod
    def _validate_dates(start_date, end_date):

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start >= end:
            raise ValueError("start_date must be before end_date")
