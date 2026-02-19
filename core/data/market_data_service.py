import logging
from pathlib import Path
import pandas as pd
import time
import numpy as np
import hashlib
import re
import tempfile
import os
import random

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

    MIN_HISTORY_ROWS = 60
    SAFE_LAG_DAYS = 2
    MAX_ROWS = 15000
    MIN_FILE_BYTES = 5_000
    MAX_DAILY_MOVE = 0.85

    _PROVIDER = None

    def __init__(self):

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        if MarketDataService._PROVIDER is None:
            MarketDataService._PROVIDER = MarketProviderRouter()

        self._fetcher = MarketDataService._PROVIDER

        self.SCHEMA_HASH = hashlib.sha256(
            ",".join(sorted(self.REQUIRED_COLUMNS)).encode()
        ).hexdigest()[:10]

    ########################################################

    @staticmethod
    def _sanitize_ticker(ticker: str):

        ticker = ticker.upper()

        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):
            raise RuntimeError(f"Unsafe ticker: {ticker}")

        return ticker

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

    @staticmethod
    def _normalize_provider_columns(df: pd.DataFrame):

        rename_map = {
            "Datetime": "date",
            "Date": "date",
            "timestamp": "date",
            "adjclose": "close",
            "Adj Close": "close",
        }

        return df.rename(columns=rename_map)

    ########################################################

    @staticmethod
    def _normalize_dates(df: pd.DataFrame):

        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="raise"
        ).dt.tz_convert(None)

        return df

    ########################################################
    # DETERMINISTIC HASH
    ########################################################

    def _dataset_hash(self, df: pd.DataFrame):

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        payload = df.to_csv(index=False).encode()

        return hashlib.sha256(payload).hexdigest()[:16]

    ########################################################

    def _atomic_write(self, df, path):

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=self.DATA_DIR,
            suffix=".tmp"
        ) as tmp:

            df.to_parquet(tmp.name, index=False)

            tmp.flush()
            os.fsync(tmp.fileno())

            temp_name = tmp.name

        os.replace(temp_name, path)

    ########################################################

    def _cache_path(self, ticker, start, end, interval):

        key = hashlib.sha256(
            f"{ticker}|{start}|{end}|{interval}|{self.SCHEMA_HASH}".encode()
        ).hexdigest()[:18]

        return self.DATA_DIR / f"{ticker}_{key}.parquet"

    ########################################################
    # INSTITUTIONAL VALIDATOR
    ########################################################

    def _validate_dataset(self, df: pd.DataFrame, ticker: str):

        if df is None or df.empty:
            raise RuntimeError("Market data empty.")

        df = self._normalize_provider_columns(df)

        df["ticker"] = ticker

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(f"Schema violation. Missing={missing}")

        df = self._normalize_dates(df)

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="raise")

        if not np.isfinite(df[numeric_cols].to_numpy()).all():
            raise RuntimeError("Non-finite prices detected.")

        if (df["close"] <= 0).any():
            raise RuntimeError("Invalid close prices.")

        # enforce strict ordering
        df = df.sort_values("date").reset_index(drop=True)

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError("Non-monotonic timestamps detected.")

        if df.duplicated(subset=["ticker", "date"]).any():
            raise RuntimeError("Duplicate (ticker,date) rows detected.")

        # safe jump detection
        pct = df["close"].pct_change().abs().fillna(0)

        if (pct > self.MAX_DAILY_MOVE).any():
            raise RuntimeError("Extreme price jump detected.")

        if len(df) > self.MAX_ROWS:
            df = df.tail(self.MAX_ROWS)

        if len(df) < self.MIN_HISTORY_ROWS:
            logger.warning(
                "Short history for %s (%d rows)",
                ticker,
                len(df)
            )

        return df.reset_index(drop=True)

    ########################################################

    def _fetch_with_retry(
        self,
        ticker,
        start,
        end,
        interval,
        retries=4
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

                validated = self._validate_dataset(df, ticker)

                validated["__dataset_hash"] = self._dataset_hash(validated)

                return validated

            except Exception as e:

                last_error = e

                sleep = (2 ** attempt) + random.uniform(0, 1)

                logger.warning(
                    "Fetch failed (%s) attempt %d/%d | sleeping %.2fs | %s",
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

    def _load_cache(self, path, ticker):

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

            expected = df["__dataset_hash"].iloc[0]

            validated = self._validate_dataset(df, ticker)

            actual = self._dataset_hash(validated)

            if expected != actual:
                logger.warning("Dataset hash mismatch — rebuilding cache.")
                path.unlink(missing_ok=True)
                return None

            return validated

        except Exception:

            logger.warning("Corrupted cache — rebuilding.")
            path.unlink(missing_ok=True)
            return None

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

        cached = self._load_cache(cache_path, ticker)

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
