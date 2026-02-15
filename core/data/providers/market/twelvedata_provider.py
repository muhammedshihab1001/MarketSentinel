import os
import logging
import requests
import pandas as pd
import numpy as np
import time

from core.data.providers.market.base import MarketDataProvider


logger = logging.getLogger(__name__)


class TwelveDataProvider(MarketDataProvider):
    """
    Institutional-grade TwelveData provider.

    Why TwelveData?
        ✔ extremely stable
        ✔ generous free tier
        ✔ clean schema
        ✔ excellent historical data
        ✔ predictable rate limits

    This provider is designed to NEVER crash training.
    """

    BASE_URL = "https://api.twelvedata.com/time_series"

    MAX_RETRIES = 3
    RETRY_SLEEP = 1.5

    MIN_ROWS = 120

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

        logger.info("TwelveData provider ready.")

    ########################################################

    def _call_api(self, params):

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                r = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=(4, 10)
                )

                r.raise_for_status()

                data = r.json()

                # TwelveData returns errors inside JSON
                if "code" in data:
                    raise RuntimeError(data.get("message"))

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

                time.sleep(self.RETRY_SLEEP)

    ########################################################

    def _normalize(self, values, ticker):

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

        ####################################################
        # SOFT invariants (do NOT over-reject)
        ####################################################

        df = df[df["high"] >= df["low"]]

        ####################################################

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(df) < self.MIN_ROWS:
            raise RuntimeError(
                f"TwelveData insufficient history ({len(df)} rows)."
            )

        df["ticker"] = ticker

        return df

    ########################################################

    def fetch(self, ticker, start_date, end_date, interval):

        interval = self.INTERVAL_MAP.get(interval, interval)

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

        df = self._normalize(data["values"], ticker)

        logger.info(
            "TwelveData served | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)
