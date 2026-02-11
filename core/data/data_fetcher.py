import yfinance as yf
import pandas as pd
import time
import hashlib
import logging
import os
import random
import requests

from datetime import datetime

logger = logging.getLogger("marketsentinel.fetcher")


class StockPriceFetcher:
    """
    Institutional Market Data Fetcher.

    Guarantees:
    - split-adjusted pricing
    - atomic cache writes
    - deterministic retries
    - gap detection
    - schema enforcement
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

    # NEW — persistent session prevents Yahoo blocking
    SESSION = requests.Session()
    SESSION.headers.update({
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        " AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/120.0 Safari/537.36"
    })

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    # =====================================================

    def _normalize_schema(self, df: pd.DataFrame):

        df = df.rename(columns=str.lower)

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True
        ).dt.tz_convert(None)

        numeric = ["open", "high", "low", "close", "volume"]

        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric)

        df["date"] = df["date"].astype("datetime64[ns]")

        return df

    # =====================================================

    def _detect_gaps(self, df):

        diffs = df["date"].diff().dt.days.dropna()

        if (diffs > self.MAX_GAP_DAYS).any():
            raise RuntimeError(
                "Large gap detected in price history."
            )

    # =====================================================

    def _validate_coverage(self, df, start_date, end_date):

        expected_days = (
            pd.to_datetime(end_date) -
            pd.to_datetime(start_date)
        ).days

        coverage = len(df) / max(expected_days, 1)

        if coverage < self.MIN_COVERAGE_RATIO:
            raise RuntimeError(
                f"Dataset coverage too low: {coverage:.2f}"
            )

    # =====================================================

    def _validate_dataset(self, df, start_date, end_date):

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

        df = df.drop_duplicates("date").sort_values("date")

        self._detect_gaps(df)
        self._validate_coverage(df, start_date, end_date)

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Dataset too small.")

        return df.reset_index(drop=True)

    # =====================================================

    def _atomic_cache_write(self, df, cache_file):

        tmp = cache_file + ".tmp"

        df.to_parquet(tmp, index=False)

        os.replace(tmp, cache_file)

    # =====================================================

    def _cache_key(self, ticker, start, end, interval):

        raw = f"{ticker}_{start}_{end}_{interval}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # =====================================================

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

        # Try cache first
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

                logger.exception(
                    "Cache corrupted — rebuilding."
                )

                os.remove(cache_file)

        # Fetch from Yahoo
        try:

            df = self._fetch_yahoo(
                ticker,
                start_date,
                end_date,
                interval
            )

        except Exception as e:

            # ⭐ FALLBACK — use stale cache if available
            if os.path.exists(cache_file):

                logger.warning(
                    f"Yahoo failed — using stale cache for {ticker}"
                )

                cached = pd.read_parquet(cache_file)

                return self._validate_dataset(
                    cached,
                    start_date,
                    end_date
                )

            raise RuntimeError(
                f"Yahoo failed and no cache available: {ticker}"
            ) from e

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
                    threads=False,
                    session=self.SESSION  #  anti-block
                )

                if df.empty:
                    raise RuntimeError(
                        "Yahoo returned empty dataframe"
                    )

                df.reset_index(inplace=True)

                return df

            except Exception as e:

                # exponential backoff + jitter
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
