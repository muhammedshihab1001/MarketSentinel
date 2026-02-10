import yfinance as yf
import pandas as pd
import time
import random
import os
import hashlib
import logging

from datetime import datetime, timedelta
from typing import List


logger = logging.getLogger("marketsentinel.fetcher")


class StockPriceFetcher:
    """
    Institutional Market Data Fetcher.

    Guarantees:
    - provider validation
    - deterministic datasets
    - corruption recovery
    - canonical schema
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
    # VALIDATION
    # =====================================================

    def _validate_dataset(self, df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Provider returned empty dataset.")

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

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Dataset too small for safe usage.")

        return df.sort_values("date")

    # =====================================================

    def _hash_dataset(self, df: pd.DataFrame) -> str:

        df = df.sort_values("date")

        data_bytes = pd.util.hash_pandas_object(
            df,
            index=True
        ).values.tobytes()

        return hashlib.sha256(data_bytes).hexdigest()

    # -----------------------------------------------------

    def _snapshot_dataset(self, ticker: str, df: pd.DataFrame):

        dataset_hash = self._hash_dataset(df)

        ticker_dir = os.path.join(
            self.DATASET_DIR,
            ticker.upper()
        )

        os.makedirs(ticker_dir, exist_ok=True)

        path = os.path.join(
            ticker_dir,
            f"{dataset_hash}.parquet"
        )

        if not os.path.exists(path):
            df.to_parquet(path, index=False)
            logger.info(f"Dataset snapshot saved → {path}")

        return dataset_hash

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

        cache_file = f"{self.CACHE_DIR}/{ticker}_{interval}.parquet"

        # ==========================================
        # LOAD CACHE
        # ==========================================

        if os.path.exists(cache_file):

            try:

                cached = pd.read_parquet(cache_file)

                cached = self._validate_dataset(cached)

                cache_start = cached["date"].min()
                cache_end = cached["date"].max()

                request_start = pd.to_datetime(start_date)
                request_end = pd.to_datetime(end_date)

                if cache_start <= request_start and cache_end >= request_end:

                    logger.info(f"Loaded fully from cache: {ticker}")

                    if snapshot:
                        self._snapshot_dataset(ticker, cached)

                    return cached

                # incremental
                missing_start = cache_end + timedelta(days=1)

                if missing_start <= request_end:

                    logger.info(
                        f"Incremental update {missing_start.date()} → {request_end.date()}"
                    )

                    new_data = self._fetch_from_providers(
                        ticker,
                        missing_start.strftime("%Y-%m-%d"),
                        end_date,
                        interval
                    )

                    new_data = self._validate_dataset(new_data)

                    combined = pd.concat(
                        [cached, new_data],
                        ignore_index=True
                    )

                    combined = self._validate_dataset(combined)

                    combined.to_parquet(cache_file, index=False)

                    if snapshot:
                        self._snapshot_dataset(ticker, combined)

                    return combined

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
                    auto_adjust=False,  # CRITICAL
                    progress=False,
                    threads=False
                )

                if df.empty:
                    raise RuntimeError("Yahoo returned empty dataframe")

                df.reset_index(inplace=True)

                df = df.rename(columns=str.lower)

                df["ticker"] = ticker

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

        df["date"] = pd.to_datetime(df["date"])

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        df = df.loc[
            (df["date"] >= start) &
            (df["date"] <= end)
        ]

        df["ticker"] = ticker

        return df

    # -----------------------------------------------------

    @staticmethod
    def _validate_dates(start_date, end_date):

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start >= end:
            raise ValueError("start_date must be before end_date")
