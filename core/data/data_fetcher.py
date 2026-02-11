import yfinance as yf
import pandas as pd
import time
import hashlib
import logging
import os
import random
from datetime import datetime

logger = logging.getLogger("marketsentinel.fetcher")


class StockPriceFetcher:
    """
    Market data fetcher with schema normalization,
    business-day validation, cache safety, and retry logic.
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
    MAX_GAP_DAYS = 10
    MIN_COVERAGE_RATIO = 0.85

    CACHE_DIR = "data/cache"

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    # -----------------------------------------------------

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten multi-index columns and remove duplicates.
        """

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]

        df = df.loc[:, ~df.columns.duplicated()]

        return df

    # -----------------------------------------------------

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize provider output into a stable schema.
        Must be idempotent.
        """

        df = self._flatten_columns(df)

        if "date" not in df.columns:
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

        df = df.loc[:, ~df.columns.duplicated()]

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            if col not in df.columns:
                raise RuntimeError(
                    f"Provider schema violation: missing={col}"
                )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(subset=["date"] + numeric_cols)

        return df

    # -----------------------------------------------------

    def _detect_gaps(self, df: pd.DataFrame):

        diffs = df["date"].diff().dt.days.dropna()

        if (diffs > self.MAX_GAP_DAYS).any():
            raise RuntimeError("Large gap detected in price history.")

    # -----------------------------------------------------

    def _validate_coverage(self, df, start_date, end_date):
        """
        Validate coverage using business days.
        """

        expected_days = len(
            pd.bdate_range(start=start_date, end=end_date)
        )

        expected_days = max(expected_days, 1)

        coverage = len(df) / expected_days

        if coverage < self.MIN_COVERAGE_RATIO:
            raise RuntimeError(
                f"Dataset coverage too low: {coverage:.2f} | "
                f"rows={len(df)} expected_business_days={expected_days}"
            )

        logger.info(
            f"Coverage OK: {coverage:.2f} ({len(df)}/{expected_days})"
        )

    # -----------------------------------------------------

    def _validate_dataset(self, df, start_date, end_date):

        if df is None or df.empty:
            raise RuntimeError("Provider returned empty dataset.")

        df = self._normalize_schema(df)

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices detected.")

        if (df["high"] < df["low"]).any():
            raise RuntimeError("High < Low detected.")

        df = df.drop_duplicates("date").sort_values("date")

        self._detect_gaps(df)
        self._validate_coverage(df, start_date, end_date)

        if len(df) < self.MIN_ROWS:
            raise RuntimeError(
                f"Dataset too small: {len(df)} rows"
            )

        return df.reset_index(drop=True)

    # -----------------------------------------------------

    def _atomic_cache_write(self, df, cache_file):

        tmp = cache_file + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, cache_file)

    # -----------------------------------------------------

    def _cache_key(self, ticker, start, end, interval):

        raw = f"{ticker}_{start}_{end}_{interval}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -----------------------------------------------------

    def fetch(self, ticker, start_date, end_date, interval="1d"):

        self._validate_dates(start_date, end_date)

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
            end_date,
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
                    raise RuntimeError(
                        "Yahoo returned empty dataframe"
                    )

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
