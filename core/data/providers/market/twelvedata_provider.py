import os
import logging
import requests
import pandas as pd
import numpy as np
import time
import threading

from core.data.providers.market.base import MarketDataProvider

logger = logging.getLogger(__name__)


class TwelveDataProvider(MarketDataProvider):

    BASE_URL = "https://api.twelvedata.com/time_series"

    MAX_RETRIES = 3
    RETRY_SLEEP = 1.5

    DEFAULT_MIN_ROWS = 120
    MAX_DAILY_MOVE = 0.90

    CALL_LOCK = threading.Lock()
    CALL_TIMESTAMPS = []

    MAX_CALLS_PER_MIN = 8
    WINDOW_SECONDS = 60

    INTERVAL_MAP = {
        "1d": "1day",
        "D": "1day",
        "1h": "1h",
        "60m": "1h",
        "15m": "15min",
        "5m": "5min",
        "1m": "1min"
    }

    ########################################################

    def __init__(self):

        self.api_key = os.getenv("TWELVEDATA_API_KEY")

        if not self.api_key:
            raise RuntimeError("TWELVEDATA_API_KEY missing.")

        self.session = requests.Session()

        logger.info("TwelveData provider ready (rate-aware).")

    ########################################################
    # RATE LIMIT ENFORCER
    ########################################################

    @classmethod
    def _respect_rate_limit(cls):

        with cls.CALL_LOCK:

            now = time.time()

            cls.CALL_TIMESTAMPS = [
                t for t in cls.CALL_TIMESTAMPS
                if now - t < cls.WINDOW_SECONDS
            ]

            if len(cls.CALL_TIMESTAMPS) >= cls.MAX_CALLS_PER_MIN:

                sleep_for = (
                    cls.WINDOW_SECONDS -
                    (now - cls.CALL_TIMESTAMPS[0])
                ) + 0.25

                logger.warning(
                    "TwelveData rate limit reached → sleeping %.2fs",
                    sleep_for
                )

                time.sleep(sleep_for)

            cls.CALL_TIMESTAMPS.append(time.time())

    ########################################################

    def _call_api(self, params):

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                self._respect_rate_limit()

                r = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=(5, 15)
                )

                r.raise_for_status()

                data = r.json()

                if "code" in data:
                    msg = data.get("message", "Vendor error")

                    if "frequency" in msg.lower():
                        logger.warning("Vendor rate limit hit — backing off.")
                        time.sleep(8)

                    raise RuntimeError(msg)

                return data

            except Exception as e:

                logger.warning(
                    "TwelveData retry %s/%s | %s",
                    attempt,
                    self.MAX_RETRIES,
                    str(e)
                )

                if attempt == self.MAX_RETRIES:
                    raise

                time.sleep(self.RETRY_SLEEP * attempt)

    ########################################################

    def _normalize(self, values, ticker, min_rows):

        df = pd.DataFrame(values)

        if df.empty:
            raise RuntimeError("TwelveData returned empty dataset.")

        df.rename(columns={"datetime": "date"}, inplace=True)

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce"
        )

        numeric = ["open", "high", "low", "close", "volume"]

        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.dropna(
            subset=["date", "open", "high", "low", "close"],
            inplace=True
        )

        if df.empty:
            raise RuntimeError("All rows invalid after normalization.")

        df = df[df["high"] >= df["low"]]

        jumps = df["close"].pct_change().abs()

        if (jumps > self.MAX_DAILY_MOVE).any():
            raise RuntimeError("Extreme price jump detected.")

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(df) < min_rows:
            raise RuntimeError(
                f"TwelveData insufficient history ({len(df)} rows)."
            )

        df["ticker"] = ticker

        return df

    ########################################################
    # PUBLIC FETCH (HARDENED)
    ########################################################

    def fetch(
        self,
        ticker,
        start_date,
        end_date,
        interval,
        **kwargs
    ):

        interval = self.INTERVAL_MAP.get(interval, interval)

        # 🔒 Safe extraction
        min_rows = kwargs.get("min_rows", self.DEFAULT_MIN_ROWS)

        params = {
            "symbol": ticker,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "outputsize": 5000,
            "apikey": self.api_key,
            "format": "JSON"
        }

        data = self._call_api(params)

        if "values" not in data:
            raise RuntimeError("TwelveData schema changed.")

        df = self._normalize(
            data["values"],
            ticker,
            min_rows
        )

        logger.info(
            "TwelveData served | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)