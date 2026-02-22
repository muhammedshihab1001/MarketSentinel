import os
import logging
import requests
import pandas as pd
import numpy as np
import time
import threading

from core.data.providers.market.base import MarketDataProvider

logger = logging.getLogger(__name__)


class AlphaVantageProvider(MarketDataProvider):

    BASE_URL = "https://www.alphavantage.co/query"

    MAX_RETRIES = 3
    RETRY_SLEEP = 12

    DEFAULT_MIN_ROWS = 120

    ##################################################
    # HARD RATE LIMIT (5/min)
    ##################################################

    CALL_LOCK = threading.Lock()
    LAST_CALL = 0
    MIN_INTERVAL = 12.5

    ##################################################

    def __init__(self):

        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY")

        if not self.api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY missing.")

        self.session = requests.Session()

        logger.info("AlphaVantage provider ready.")

    ##################################################

    @classmethod
    def _respect_rate_limit(cls):

        with cls.CALL_LOCK:

            now = time.time()
            elapsed = now - cls.LAST_CALL

            if elapsed < cls.MIN_INTERVAL:

                sleep_for = cls.MIN_INTERVAL - elapsed

                logger.warning(
                    "AlphaVantage throttle → sleeping %.2fs",
                    sleep_for
                )

                time.sleep(sleep_for)

            cls.LAST_CALL = time.time()

    ##################################################

    def _call_api(self, symbol):

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key
        }

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                self._respect_rate_limit()

                r = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=(6, 20)
                )

                r.raise_for_status()

                data = r.json()

                if "Error Message" in data:
                    raise RuntimeError(data["Error Message"])

                if "Note" in data:
                    logger.warning("AlphaVantage rate note received.")
                    time.sleep(20)
                    raise RuntimeError("Rate limited")

                return data

            except Exception as e:

                logger.warning(
                    "AlphaVantage retry %s/%s | %s",
                    attempt,
                    self.MAX_RETRIES,
                    str(e)
                )

                if attempt == self.MAX_RETRIES:
                    raise

                time.sleep(self.RETRY_SLEEP)

    ##################################################

    def _normalize(self, data, ticker, start_date, min_rows):

        key = "Time Series (Daily)"

        if key not in data:
            raise RuntimeError("AlphaVantage schema changed.")

        df = pd.DataFrame.from_dict(
            data[key],
            orient="index"
        ).reset_index()

        df.rename(columns={
            "index": "date",
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "6. volume": "volume"
        }, inplace=True)

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce"
        )

        df = df[df["date"] >= pd.Timestamp(start_date, tz="UTC")]

        numeric = ["open", "high", "low", "close", "volume"]

        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.dropna(
            subset=["open", "high", "low", "close"],
            inplace=True
        )

        df = df[df["high"] >= df["low"]]

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(df) < min_rows:
            raise RuntimeError(
                f"AlphaVantage insufficient history ({len(df)} rows)."
            )

        df["ticker"] = ticker

        return df

    ##################################################
    # PUBLIC FETCH (HARDENED)
    ##################################################

    def fetch(
        self,
        ticker,
        start_date,
        end_date,
        interval,
        **kwargs
    ):

        if interval not in ["1d", "D"]:
            raise RuntimeError(
                "AlphaVantage supports daily only."
            )

        min_rows = kwargs.get("min_rows", self.DEFAULT_MIN_ROWS)

        data = self._call_api(ticker)

        df = self._normalize(
            data=data,
            ticker=ticker,
            start_date=start_date,
            min_rows=min_rows
        )

        logger.info(
            "AlphaVantage served | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)