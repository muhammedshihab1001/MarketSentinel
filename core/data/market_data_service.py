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

    DEFAULT_MIN_HISTORY_ROWS = 60   # 🔥 configurable
    SAFE_LAG_DAYS = 2
    MAX_ROWS = 15000
    MIN_FILE_BYTES = 5_000
    MAX_DAILY_MOVE = 0.85

    _PROVIDER = None

    ########################################################

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

    def _dataset_hash(self, df: pd.DataFrame):

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        payload = df.to_csv(index=False).encode()

        return hashlib.sha256(payload).hexdigest()[:16]

    ########################################################

    def _validate_dataset(self, df: pd.DataFrame, ticker: str, min_rows: int):

        if df is None or df.empty:
            raise RuntimeError("Market data empty.")

        df = df.copy()
        df["ticker"] = ticker

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(f"Schema violation. Missing={missing}")

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="raise")

        if not np.isfinite(df[numeric_cols].to_numpy()).all():
            raise RuntimeError("Non-finite prices detected.")

        df = df.sort_values("date").reset_index(drop=True)

        pct = df["close"].pct_change().abs().fillna(0)

        if (pct > self.MAX_DAILY_MOVE).any():
            raise RuntimeError("Extreme price jump detected.")

        if len(df) > self.MAX_ROWS:
            df = df.tail(self.MAX_ROWS)

        # 🔥 configurable tolerance
        if len(df) < min_rows:
            raise RuntimeError(
                f"Insufficient history ({len(df)} < {min_rows})"
            )

        return df.reset_index(drop=True)

    ########################################################

    def _fetch_with_retry(
        self,
        ticker,
        start,
        end,
        interval,
        min_history,
        retries=4
    ):

        last_error = None

        for attempt in range(retries):

            try:

                df = self._fetcher.fetch(
                    ticker,
                    start,
                    end,
                    interval,
                    min_rows=min_history  # 🔥 propagate
                )

                validated = self._validate_dataset(
                    df,
                    ticker,
                    min_history
                )

                validated["__dataset_hash"] = self._dataset_hash(validated)

                return validated

            except Exception as e:

                last_error = e

                sleep = (2 ** attempt) + random.uniform(0, 1)

                logger.warning(
                    "Fetch failed (%s) attempt %d/%d | %.2fs | %s",
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

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        min_history: int | None = None   # 🔥 NEW PARAM
    ):

        ticker = self._sanitize_ticker(ticker)

        end_date = self._cap_to_safe_date(end_date)

        if min_history is None:
            min_history = self.DEFAULT_MIN_HISTORY_ROWS

        logger.info(
            "Fetching dataset for %s (min_history=%d)",
            ticker,
            min_history
        )

        df = self._fetch_with_retry(
            ticker,
            start_date,
            end_date.strftime("%Y-%m-%d"),
            interval,
            min_history
        )

        return df.reset_index(drop=True)