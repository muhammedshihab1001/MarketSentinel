import yfinance as yf
import pandas as pd
import time
import hashlib
import logging
import os
import random
from datetime import timedelta
import uuid
import numpy as np


logger = logging.getLogger(__name__)


class StockPriceFetcher:

    REQUIRED_COLUMNS = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    CACHE_VERSION = "6.0"

    MIN_ROWS = 120
    MAX_RETRIES = 5
    BASE_SLEEP = 1.5
    MAX_GAP_DAYS = 10

    CRITICAL_COVERAGE = 0.55
    WARNING_COVERAGE = 0.70

    MAX_ZERO_VOLUME_RATIO = 0.20
    MAX_DAILY_RETURN = 1.50

    CACHE_DIR = "data/cache"

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    @staticmethod
    def _fsync_dir(directory):

        if os.name == "nt":
            return

        fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _to_utc(ts):
        ts = pd.Timestamp(ts)

        if ts.tzinfo is None:
            return ts.tz_localize("UTC")

        return ts.tz_convert("UTC")

    def _cap_to_yesterday(self, end_date: str):

        requested = self._to_utc(end_date)

        yesterday = (
            pd.Timestamp.utcnow()
            .tz_localize("UTC")
            - timedelta(days=1)
        )

        return min(requested, yesterday)

    @staticmethod
    def _validate_dates(start_date, end_date):

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        if start >= end:
            raise ValueError("start_date must be before end_date")

    def _flatten_columns(self, df: pd.DataFrame):

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]

        return df.loc[:, ~df.columns.duplicated()]

    def _detect_datetime_column(self, df):

        for col in df.columns:
            if "date" in col or "time" in col:
                return col

        raise RuntimeError("No datetime column detected.")

    def _normalize_schema(self, df: pd.DataFrame, ticker: str):

        df = self._flatten_columns(df)

        if "adj close" in df.columns:
            df["close"] = df["adj close"]

        if "close" not in df.columns:
            raise RuntimeError("Close column missing from provider.")

        if "date" not in df.columns:
            dt_col = self._detect_datetime_column(df)
            df.rename(columns={dt_col: "date"}, inplace=True)

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="raise"
        )

        numeric_cols = ["open", "high", "low", "close", "volume"]

        missing = [c for c in numeric_cols if c not in df.columns]

        if missing:
            raise RuntimeError(
                f"Provider schema violation after normalization: {missing}"
            )

        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("float64")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["date"] + numeric_cols)

        df = df[["date", "open", "high", "low", "close", "volume"]]

        df["ticker"] = ticker

        return df[self.REQUIRED_COLUMNS]

    def _coverage_ratio(self, df, start_date, end_date):

        expected = pd.date_range(
            start=start_date,
            end=end_date,
            freq="B"
        )

        coverage = len(df) / max(len(expected), 1)

        if coverage < self.CRITICAL_COVERAGE:
            raise RuntimeError("Critical coverage failure.")

        if coverage < self.WARNING_COVERAGE:
            logger.warning("Low coverage detected: %.2f", coverage)

    def _validate_dataset(self, df, start_date, end_date, ticker):

        if df is None or df.empty:
            raise RuntimeError("Provider returned empty dataset.")

        df = self._normalize_schema(df, ticker)

        df = df.drop_duplicates(["ticker", "date"]).sort_values("date")

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Non-monotonic timestamps detected.")

        returns = df["close"].pct_change().abs()

        if (returns > self.MAX_DAILY_RETURN).any():
            logger.warning("Extreme return detected for %s", ticker)

        zero_ratio = (df["volume"] == 0).mean()

        if zero_ratio > self.MAX_ZERO_VOLUME_RATIO:
            logger.warning("High zero-volume ratio for %s", ticker)

        if len(df) < self.MIN_ROWS:
            raise RuntimeError(f"Dataset too small: {len(df)} rows")

        self._coverage_ratio(df, start_date, end_date)

        return df.reset_index(drop=True)

    def _atomic_cache_write(self, df, cache_file):

        directory = os.path.dirname(cache_file) or "."
        tmp = f"{cache_file}.{uuid.uuid4().hex}.tmp"

        df.to_parquet(tmp, index=False)

        # verify
        pd.read_parquet(tmp)

        os.replace(tmp, cache_file)
        self._fsync_dir(directory)

    def _cache_key(self, ticker, start, end, interval):

        raw = f"{self.CACHE_VERSION}|{ticker}|{start}|{end}|{interval}|schema=v2"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def fetch(self, ticker, start_date, end_date, interval="1d"):

        self._validate_dates(start_date, end_date)
        end_date = self._cap_to_yesterday(end_date)

        cache_file = (
            f"{self.CACHE_DIR}/"
            f"{self._cache_key(ticker, start_date, end_date, interval)}.parquet"
        )

        if os.path.exists(cache_file):

            try:
                cached = pd.read_parquet(cache_file)

                return self._validate_dataset(
                    cached,
                    start_date,
                    end_date,
                    ticker
                )

            except Exception:
                logger.exception("Cache corrupted. Rebuilding.")
                os.remove(cache_file)

        df = self._fetch_yahoo(
            ticker,
            start_date,
            end_date,
            interval
        )

        df = self._validate_dataset(
            df,
            start_date,
            end_date,
            ticker
        )

        self._atomic_cache_write(df, cache_file)

        return df

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

                return df

            except Exception as e:

                sleep_time = (
                    self.BASE_SLEEP * (2 ** attempt)
                    + random.uniform(0, 1)
                )

                logger.warning(
                    "Yahoo retry %s/%s in %.2fs | %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    sleep_time,
                    str(e)
                )

                time.sleep(sleep_time)

        raise RuntimeError("Yahoo failed after retries.")
