import yfinance as yf
import pandas as pd
import time
import hashlib
import logging
import os
import random

from datetime import datetime, timedelta


logger = logging.getLogger("marketsentinel.fetcher")


class StockPriceFetcher:

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
    CACHE_VERSION = "2.0"

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    # -----------------------------------------------------

    @staticmethod
    def _cap_to_yesterday(date_str):

        requested = pd.Timestamp(date_str).normalize()

        yesterday = (
            pd.Timestamp.utcnow()
            .normalize() - pd.Timedelta(days=1)
        )

        return min(requested, yesterday)

    # -----------------------------------------------------

    def _cache_key(self, ticker, start, end, interval):

        raw = (
            f"{ticker}|{start}|{end}|{interval}|"
            f"adj=auto|schema=v2|cache={self.CACHE_VERSION}"
        )

        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    # -----------------------------------------------------

    def _atomic_cache_write(self, df, cache_file):

        tmp = cache_file + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, cache_file)

    # -----------------------------------------------------

    def _normalize_schema(self, df: pd.DataFrame):

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]

        if "date" not in df.columns:
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:

            if col not in df.columns:
                raise RuntimeError(f"Provider schema violation: {col}")

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["date"] + numeric_cols, inplace=True)

        return df

    # -----------------------------------------------------

    def _validate_dataset(self, df, start_date, end_date):

        df = self._normalize_schema(df)

        df = df.drop_duplicates("date").sort_values("date")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if len(df) < self.MIN_ROWS:
            raise RuntimeError(f"Dataset too small: {len(df)} rows")

        return df.reset_index(drop=True)

    # -----------------------------------------------------

    def fetch(self, ticker, start_date, end_date, interval="1d"):

        end_date = self._cap_to_yesterday(end_date)

        cache_file = (
            f"{self.CACHE_DIR}/"
            f"{self._cache_key(ticker, start_date, end_date, interval)}.parquet"
        )

        if os.path.exists(cache_file):

            try:
                cached = pd.read_parquet(cache_file)
                logger.info(f"Cache hit: {ticker}")

                return self._validate_dataset(
                    cached,
                    start_date,
                    end_date
                )

            except Exception:
                logger.exception("Cache corrupted. Rebuilding.")
                os.remove(cache_file)

        df = self._fetch_yahoo(
            ticker,
            start_date,
            end_date.strftime("%Y-%m-%d"),
            interval
        )

        df = self._validate_dataset(
            df,
            start_date,
            end_date
        )

        self._atomic_cache_write(df, cache_file)

        logger.info(f"Cached dataset: {ticker}")

        return df

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
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )

                if df.empty:
                    raise RuntimeError("Yahoo returned empty dataframe")

                return df

            except Exception as e:

                sleep_time = (
                    self.BASE_SLEEP * (2 ** attempt)
                    + random.uniform(0, 1)
                )

                logger.warning(
                    f"Yahoo retry {attempt+1}/{self.MAX_RETRIES} "
                    f"in {round(sleep_time,2)}s: {e}"
                )

                time.sleep(sleep_time)

        raise RuntimeError("Yahoo failed after retries.")

    # -----------------------------------------------------

    @staticmethod
    def _validate_dates(start_date, end_date):

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start >= end:
            raise ValueError("start_date must be before end_date")
