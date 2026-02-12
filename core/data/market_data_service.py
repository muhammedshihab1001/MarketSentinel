import logging
from pathlib import Path
import pandas as pd
import time
import os
import numpy as np
import hashlib
import re

from core.data.data_fetcher import StockPriceFetcher


logger = logging.getLogger("marketsentinel.market_data")


class MarketDataService:
    """
    Institutional Market Data Layer.

    Guarantees:
    ✔ zero lookahead bias
    ✔ schema drift detection
    ✔ concurrent write protection
    ✔ corruption recovery
    ✔ ticker sanitization
    ✔ memory safety
    ✔ atomic durability
    """

    DATA_DIR = Path("data/lake")

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    MIN_HISTORY_ROWS = 120
    REVISION_DAYS = 5
    SAFE_LAG_DAYS = 2
    MAX_FILES = 400

    MAX_ROWS = 15000   # prevents runaway merges

    ########################################################

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._fetcher = StockPriceFetcher()

        self.SCHEMA_HASH = hashlib.sha256(
            ",".join(sorted(self.REQUIRED_COLUMNS)).encode()
        ).hexdigest()[:10]

    ########################################################
    # TICKER SAFETY
    ########################################################

    @staticmethod
    def _sanitize_ticker(ticker: str):

        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):
            raise RuntimeError(f"Unsafe ticker: {ticker}")

        return ticker

    ########################################################

    def _dataset_path(self, ticker: str, interval: str):

        ticker = self._sanitize_ticker(ticker)

        name = f"{ticker}_{interval}_{self.SCHEMA_HASH}.parquet"

        return self.DATA_DIR / name

    ########################################################
    # LOCKING
    ########################################################

    def _acquire_lock(self, path: Path):

        lock = path.with_suffix(".lock")

        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock

        except FileExistsError:
            raise RuntimeError(
                f"Concurrent write detected for {path.name}"
            )

    @staticmethod
    def _release_lock(lock: Path):

        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass

    ########################################################
    # LOOKAHEAD PROTECTION
    ########################################################

    @classmethod
    def _cap_to_safe_date(cls, date_str: str):

        requested = pd.Timestamp(date_str).normalize()

        safe_cutoff = (
            pd.Timestamp.utcnow().normalize()
            - pd.Timedelta(days=cls.SAFE_LAG_DAYS)
        )

        return min(requested, safe_cutoff)

    ########################################################
    # HARD VALIDATOR
    ########################################################

    def _validate_dataset(self, df: pd.DataFrame):

        if df is None or df.empty:
            raise RuntimeError("Market data empty.")

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Schema violation. Missing={missing}"
            )

        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce"
        ).dt.tz_convert(None)

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
            raise RuntimeError("Duplicate timestamps.")

        if not df["date"].is_monotonic_increasing:
            df = df.sort_values("date")

        if len(df) < self.MIN_HISTORY_ROWS:
            raise RuntimeError("Insufficient market history.")

        if len(df) > self.MAX_ROWS:
            logger.warning(
                "Dataset too large — trimming oldest rows."
            )
            df = df.tail(self.MAX_ROWS)

        return df.reset_index(drop=True)

    ########################################################
    # ATOMIC WRITE
    ########################################################

    def _atomic_save(self, df: pd.DataFrame, path: Path):

        lock = self._acquire_lock(path)

        try:

            tmp = path.with_suffix(".tmp")

            df.to_parquet(tmp, index=False)

            with open(tmp, "rb+") as f:
                f.flush()
                os.fsync(f.fileno())

            tmp.replace(path)

            if os.name != "nt":
                fd = os.open(str(path.parent), os.O_DIRECTORY)
                os.fsync(fd)
                os.close(fd)

        finally:
            self._release_lock(lock)

    ########################################################
    # DISK CONTROL
    ########################################################

    def _prune_disk(self):

        files = sorted(
            self.DATA_DIR.glob("*.parquet"),
            key=lambda x: x.stat().st_mtime
        )

        if len(files) <= self.MAX_FILES:
            return

        for f in files[:50]:
            try:
                f.unlink()
            except Exception:
                pass

    ########################################################

    def _load_local(self, path: Path):

        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            return self._validate_dataset(df)

        except Exception:

            logger.exception(
                "Local dataset corrupted — rebuilding."
            )

            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

            return None

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

                return self._validate_dataset(df)

            except Exception as e:

                last_error = e

                logger.warning(
                    "Fetch failed (%s) attempt %d/%d",
                    ticker,
                    attempt + 1,
                    retries
                )

                time.sleep(1.5)

        raise RuntimeError(
            f"Market fetch failed after retries: {ticker}"
        ) from last_error

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

        path = self._dataset_path(ticker, interval)

        local_df = self._load_local(path)

        ####################################################
        # FIRST BUILD
        ####################################################

        if local_df is None:

            logger.info("Building dataset for %s", ticker)

            df = self._fetch_with_retry(
                ticker,
                start_date,
                end_date.strftime("%Y-%m-%d"),
                interval
            )

            self._atomic_save(df, path)
            self._prune_disk()

            return df

        ####################################################
        # REVISION WINDOW
        ####################################################

        revision_start = (
            pd.to_datetime(local_df["date"].max())
            - pd.Timedelta(days=self.REVISION_DAYS)
        ).strftime("%Y-%m-%d")

        try:

            revision_df = self._fetch_with_retry(
                ticker,
                revision_start,
                end_date.strftime("%Y-%m-%d"),
                interval
            )

            local_df = pd.concat(
                [local_df, revision_df],
                ignore_index=True
            )

        except Exception:
            logger.exception(
                "Revision fetch failed — using cached data."
            )

        local_df = (
            local_df
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

        self._atomic_save(local_df, path)
        self._prune_disk()

        mask = (
            (local_df["date"] >= pd.Timestamp(start_date)) &
            (local_df["date"] <= end_date)
        )

        return local_df.loc[mask].reset_index(drop=True)
