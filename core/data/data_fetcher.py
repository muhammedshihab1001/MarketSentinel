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
    Institutional Market Data Fetcher.

    Guarantees:
    schema normalization
    business-day coverage validation
    atomic cache writes
    retry with jitter
    gap detection
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

    # Institutional tolerance
    MIN_COVERAGE_RATIO = 0.85

    CACHE_DIR = "data/cache"

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    # -----------------------------------------------------

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Yahoo multi-index columns.
        """

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        return df

    # -----------------------------------------------------

    def _normalize_schema(self, df: pd.DataFrame):
        """
        Force provider output into institutional schema.
        """

        df = self._flatten_columns(df)

        # Force index → date column safely
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True
        ).dt.tz_convert(None)

        numeric = ["open", "high", "low", "close", "volume"]

        for col in numeric:

            if col not in df.columns:
                raise RuntimeError(
                    f"Provider schema violation: missing={col}"
                )

            df[col] = pd.to_numeric(
                df[col].squeeze(),
                errors="coerce"
            )

        df = df.dropna(subset=numeric)

        return df

    # -----------------------------------------------------

    def _detect_gaps(self, df):
        """
        Detect abnormal missing ranges.
        """

        diffs = df["date"].diff().dt.days.dropna()

        if (diffs > self.MAX_GAP_DAYS).any():
            raise RuntimeError(
                "Large gap detected in price history."
            )

    # -----------------------------------------------------
    # FIXED — BUSINESS DAY COVERAGE 
    # -----------------------------------------------------

    def _validate_coverage(self, df, start_date, end_date):
        """
        Validate dataset coverage using BUSINESS DAYS.

        Stocks do not trade on weekends.
        Calendar-day validation is incorrect.
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
            f"Coverage OK → {coverage:.2f} "
            f"({len(df)}/{expected_days})"
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

    def fetch(
        self,
        ticker,
        start_date,
        end_date,
        interval="1d"
    ):

        self._validate_dates(start_date, end_date)

        cache_name = self._cache_key(
            ticker,
            start_date,
            end_date,
            interval
        )

        cache_file = f"{self.CACHE_DIR}/{cache_name}.parquet"

        if os.path.exists(cache_file):

            try:
                cached = pd.read_parquet(cache_file)
                logger.info(f"Cache hit → {ticker}")

                return self._validate_dataset(
                    cached,
                    start_date,
                    end_date
                )

            except Exception:
                logger.exception("Cache corrupted — rebuilding.")
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

        logger.info(f"Cached dataset → {ticker}")

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
                    f"[Yahoo] retry {attempt+1}/{self.MAX_RETRIES} "
                    f"in {round(sleep_time,2)}s → {e}"
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
