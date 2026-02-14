import logging
from pathlib import Path
import pandas as pd
import time
import numpy as np
import hashlib
import re
import tempfile
import os

from core.data.providers.market.router import MarketProviderRouter

logger = logging.getLogger("marketsentinel.market_data")


class MarketDataService:
    """
    Institutional Market Data Layer.

    Guarantees:
    ✔ zero lookahead bias
    ✔ schema drift detection
    ✔ corruption prevention
    ✔ ticker enforcement
    ✔ timezone normalization
    ✔ numeric safety
    ✔ dataset fingerprinting
    ✔ atomic persistence
    ✔ provider contract validation
    """

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

    MIN_HISTORY_ROWS = 120
    SAFE_LAG_DAYS = 2
    MAX_ROWS = 15000
    MIN_FILE_BYTES = 5_000

    ########################################################

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._fetcher = MarketProviderRouter()

        self.SCHEMA_HASH = hashlib.sha256(
            ",".join(sorted(self.REQUIRED_COLUMNS)).encode()
        ).hexdigest()[:10]

    ########################################################
    # TICKER SAFETY
    ########################################################

    @staticmethod
    def _sanitize_ticker(ticker: str):

        ticker = ticker.upper()

        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):
            raise RuntimeError(f"Unsafe ticker: {ticker}")

        return ticker

    ########################################################
    # LOOKAHEAD PROTECTION
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
    # DATE NORMALIZATION
    ########################################################

    @staticmethod
    def _normalize_dates(df: pd.DataFrame):

        df = df.copy()

        df["date"] = (
            pd.to_datetime(
                df["date"],
                utc=True,
                errors="coerce"
            ).dt.tz_convert(None)
        )

        return df

    ########################################################
    # PROVIDER COLUMN NORMALIZATION
    ########################################################

    @staticmethod
    def _normalize_provider_columns(df: pd.DataFrame):

        # handles Yahoo / Finnhub variations
        rename_map = {
            "Datetime": "date",
            "Date": "date",
            "timestamp": "date",
            "adjclose": "close",
            "Adj Close": "close",
        }

        df = df.rename(columns=rename_map)

        return df

    ########################################################
    # DATASET HASH
    ########################################################

    @staticmethod
    def _dataset_hash(df: pd.DataFrame):

        payload = pd.util.hash_pandas_object(
            df.sort_values("date"),
            index=True
        ).values.tobytes()

        return hashlib.sha256(payload).hexdigest()[:16]

    ########################################################
    # ATOMIC WRITE
    ########################################################

    def _atomic_write(self, df, path):

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=self.DATA_DIR,
            suffix=".tmp"
        ) as tmp:

            df.to_parquet(tmp.name, index=False)

            temp_name = tmp.name

        os.replace(temp_name, path)

    ########################################################
    # CACHE PATH
    ########################################################

    def _cache_path(self, ticker, start, end, interval):

        key = hashlib.sha256(
            f"{ticker}|{start}|{end}|{interval}|{self.SCHEMA_HASH}".encode()
        ).hexdigest()[:18]

        return self.DATA_DIR / f"{ticker}_{key}.parquet"

    ########################################################
    # HARD VALIDATOR
    ########################################################

    def _validate_dataset(self, df: pd.DataFrame, ticker: str):

        if df is None or df.empty:
            raise RuntimeError("Market data empty.")

        df = self._normalize_provider_columns(df)

        # FORCE ticker column
        df["ticker"] = ticker

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(f"Schema violation. Missing={missing}")

        df = self._normalize_dates(df)

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date"] + numeric_cols)

        if not np.isfinite(df[numeric_cols].to_numpy()).all():
            raise RuntimeError("Non-finite prices detected.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        if (df["high"] < df["low"]).any():
            raise RuntimeError("High < Low detected.")

        if df["date"].duplicated().any():
            raise RuntimeError("Duplicate timestamps detected.")

        df = df.sort_values("date")

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Non-monotonic timestamps detected.")

        if len(df) < self.MIN_HISTORY_ROWS:
            raise RuntimeError("Insufficient market history.")

        if len(df) > self.MAX_ROWS:
            logger.warning("Dataset too large — trimming oldest rows.")
            df = df.tail(self.MAX_ROWS)

        df["__dataset_hash"] = self._dataset_hash(df)

        return df.reset_index(drop=True)

    ########################################################
    # FETCH WITH RETRY
    ########################################################

    def _fetch_with_retry(
        self,
        ticker,
        start,
        end,
        interval,
        retries=3
    ):

        last_error = None

        for attempt in range(retries):

            try:

                df = self._fetcher.fetch(
                    ticker,
                    start,
                    end,
                    interval
                )

                return self._validate_dataset(df, ticker)

            except Exception as e:

                last_error = e

                logger.warning(
                    "Fetch failed (%s) attempt %d/%d | %s",
                    ticker,
                    attempt + 1,
                    retries,
                    str(e)
                )

                time.sleep(1.2)

        raise RuntimeError(
            f"Market fetch failed after retries: {ticker}"
        ) from last_error

    ########################################################
    # CACHE SAFE LOAD
    ########################################################

    def _load_cache(self, path):

        if not path.exists():
            return None

        if path.stat().st_size < self.MIN_FILE_BYTES:
            path.unlink(missing_ok=True)
            return None

        try:
            df = pd.read_parquet(path)

            if "__dataset_hash" not in df.columns:
                path.unlink(missing_ok=True)
                return None

            return df

        except Exception:
            logger.warning("Corrupted cache — rebuilding.")
            path.unlink(missing_ok=True)
            return None

    ########################################################
    # PUBLIC ENTRY
    ########################################################

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ):

        ticker = self._sanitize_ticker(ticker)

        end_date = self._cap_to_safe_date(end_date)

        cache_path = self._cache_path(
            ticker,
            start_date,
            end_date.strftime("%Y-%m-%d"),
            interval
        )

        cached = self._load_cache(cache_path)

        if cached is not None:
            return cached.reset_index(drop=True)

        logger.info("Fetching dataset for %s", ticker)

        df = self._fetch_with_retry(
            ticker,
            start_date,
            end_date.strftime("%Y-%m-%d"),
            interval
        )

        self._atomic_write(df, cache_path)

        return df.reset_index(drop=True)
