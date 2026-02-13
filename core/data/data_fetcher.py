import yfinance as yf
import pandas as pd
import time
import hashlib
import logging
import os
import random
from datetime import timedelta
import uuid


logger = logging.getLogger("marketsentinel.fetcher")


class StockPriceFetcher:

    REQUIRED_COLUMNS = {
        "ticker",   # ⭐ NOW HARD REQUIRED
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    CACHE_VERSION = "3.0"  # ⭐ bump version to invalidate old cache

    MIN_ROWS = 120
    MAX_RETRIES = 5
    BASE_SLEEP = 1.5
    MAX_GAP_DAYS = 10

    CRITICAL_COVERAGE = 0.55
    WARNING_COVERAGE = 0.70

    MAX_ZERO_VOLUME_RATIO = 0.05
    MAX_DAILY_RETURN = 0.60

    CACHE_DIR = "data/cache"

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    ##################################################
    # FSYNC
    ##################################################

    @staticmethod
    def _fsync_dir(directory):

        if os.name == "nt":
            return

        fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    ##################################################
    # TIME SAFETY
    ##################################################

    @staticmethod
    def _to_naive_utc(ts):

        ts = pd.Timestamp(ts)

        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)

        return ts

    def _cap_to_yesterday(self, end_date: str):

        requested = self._to_naive_utc(end_date)

        yesterday = (
            pd.Timestamp.utcnow()
            .tz_localize(None)
            - timedelta(days=1)
        )

        return min(requested, yesterday)

    ##################################################

    @staticmethod
    def _validate_dates(start_date, end_date):

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        if start >= end:
            raise ValueError("start_date must be before end_date")

    ##################################################

    def _flatten_columns(self, df: pd.DataFrame):

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]

        return df.loc[:, ~df.columns.duplicated()]

    ##################################################
    # NORMALIZE + ATTACH TICKER (CRITICAL FIX)
    ##################################################

    def _normalize_schema(self, df: pd.DataFrame, ticker: str):

        df = self._flatten_columns(df)

        if "close" not in df.columns and "adj close" in df.columns:
            df.rename(columns={"adj close": "close"}, inplace=True)

        if "adj close" in df.columns:
            df["close"] = df["adj close"]
            df.drop(columns=["adj close"], inplace=True)

        if "date" not in df.columns:
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce", utc=True)
            .dt.tz_convert(None)
        )

        numeric_cols = ["open", "high", "low", "close", "volume"]

        missing = [c for c in numeric_cols if c not in df.columns]

        if missing:
            raise RuntimeError(
                f"Provider schema violation after normalization: {missing}"
            )

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date"] + numeric_cols)

        df = df.loc[:, ["date", "open", "high", "low", "close", "volume"]]

        ##################################################
        # ⭐ THE FIX THAT STOPS YOUR TRAINING CRASH
        ##################################################

        df["ticker"] = ticker

        return df

    ##################################################
    # HARD VALIDATION
    ##################################################

    def _detect_gaps(self, df: pd.DataFrame):

        diffs = df["date"].diff().dt.days.dropna()

        if (diffs > self.MAX_GAP_DAYS).any():
            raise RuntimeError("Large gap detected in price history.")

    def _validate_monotonic(self, df):

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Non-monotonic timestamps detected.")

    def _validate_returns(self, df):

        returns = df["close"].pct_change().abs()

        if (returns > self.MAX_DAILY_RETURN).any():
            raise RuntimeError(
                "Extreme price spike detected — possible bad data."
            )

    def _validate_volume(self, df):

        zero_ratio = (df["volume"] == 0).mean()

        if zero_ratio > self.MAX_ZERO_VOLUME_RATIO:
            raise RuntimeError(
                "Too many zero-volume rows — provider unreliable."
            )

    def _validate_coverage(self, df, start_date, end_date):

        start = self._to_naive_utc(start_date)
        end = self._to_naive_utc(end_date)

        expected_sessions = len(
            pd.bdate_range(start=start, end=end)
        )

        expected_sessions = max(expected_sessions, 1)

        coverage = len(df) / expected_sessions

        if coverage < self.CRITICAL_COVERAGE:
            raise RuntimeError(
                f"Dataset coverage critically low: {coverage:.2f}"
            )

        if coverage < self.WARNING_COVERAGE:
            logger.warning(
                f"Low dataset coverage ({coverage:.2f}). Continuing."
            )

    ##################################################

    def _validate_dataset(self, df, start_date, end_date, ticker):

        if df is None or df.empty:
            raise RuntimeError("Provider returned empty dataset.")

        df = self._normalize_schema(df, ticker)

        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            raise RuntimeError("Date column corrupted.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if (df["high"] < df["low"]).any():
            raise RuntimeError("High < Low detected.")

        df = df.drop_duplicates(["ticker", "date"]).sort_values("date")

        self._validate_monotonic(df)
        self._detect_gaps(df)
        self._validate_returns(df)
        self._validate_volume(df)
        self._validate_coverage(df, start_date, end_date)

        if len(df) < self.MIN_ROWS:
            raise RuntimeError(
                f"Dataset too small: {len(df)} rows"
            )

        return df.reset_index(drop=True)

    ##################################################
    # CACHE
    ##################################################

    def _atomic_cache_write(self, df, cache_file):

        directory = os.path.dirname(cache_file) or "."
        tmp = f"{cache_file}.{uuid.uuid4().hex}.tmp"

        df.to_parquet(tmp, index=False)

        os.replace(tmp, cache_file)
        self._fsync_dir(directory)

    def _cache_key(self, ticker, start, end, interval):

        raw = f"{self.CACHE_VERSION}_{ticker}_{start}_{end}_{interval}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    ##################################################
    # PUBLIC FETCH
    ##################################################

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
                try:
                    os.remove(cache_file)
                except Exception:
                    pass

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

    ##################################################

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
                    f"Yahoo retry {attempt+1}/{self.MAX_RETRIES} "
                    f"in {round(sleep_time,2)}s: {e}"
                )

                time.sleep(sleep_time)

        raise RuntimeError("Yahoo failed after retries.")
