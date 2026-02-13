import os
import logging
import requests
import pandas as pd

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.data.providers.market.base import MarketDataProvider


logger = logging.getLogger("marketsentinel.provider.finnhub")


class FinnhubProvider(MarketDataProvider):

    URL = "https://finnhub.io/api/v1/stock/candle"

    REQUEST_TIMEOUT = (4, 15)

    def __init__(self):

        self.key = os.getenv("FINNHUB_API_KEY")

        if not self.key:
            raise RuntimeError("FINNHUB_API_KEY missing.")

        self.session = self._build_session()

    ########################################################

    def _build_session(self):

        session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=retry
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update({
            "User-Agent": "MarketSentinel/Institutional"
        })

        return session

    ########################################################

    @staticmethod
    def _to_unix(dt: str):
        return int(pd.Timestamp(dt).timestamp())

    ########################################################

    def _normalize_schema(self, payload, ticker):

        df = pd.DataFrame({
            "date": pd.to_datetime(payload["t"], unit="s", utc=True)
                    .tz_convert(None),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"]
        })

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date"] + numeric_cols)

        if df.empty:
            raise RuntimeError("Finnhub returned empty normalized dataset.")

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        df["ticker"] = ticker

        return df

    ########################################################

    def fetch(self, ticker, start_date, end_date, interval="1d"):

        resolution = {
            "1d": "D",
            "1h": "60",
            "5m": "5"
        }.get(interval, "D")

        params = {
            "symbol": ticker,
            "resolution": resolution,
            "from": self._to_unix(start_date),
            "to": self._to_unix(end_date),
            "token": self.key
        }

        response = self.session.get(
            self.URL,
            params=params,
            timeout=self.REQUEST_TIMEOUT
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("s") != "ok":
            raise RuntimeError(
                f"Finnhub returned no usable data for {ticker}"
            )

        df = self._normalize_schema(payload, ticker)

        logger.debug(
            "Finnhub served market data | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return df
