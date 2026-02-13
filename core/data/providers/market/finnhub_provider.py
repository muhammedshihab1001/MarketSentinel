import os
import requests
import pandas as pd
from datetime import datetime

from core.data.providers.market.base import MarketDataProvider


class FinnhubProvider(MarketDataProvider):

    URL = "https://finnhub.io/api/v1/stock/candle"

    def __init__(self):
        self.key = os.getenv("FINNHUB_API_KEY")

        if not self.key:
            raise RuntimeError("FINNHUB_API_KEY missing.")

        self.session = requests.Session()

    def _to_unix(self, dt: str):
        return int(pd.Timestamp(dt).timestamp())

    def fetch(self, ticker, start_date, end_date, interval="D"):

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

        r = self.session.get(self.URL, params=params, timeout=20)
        r.raise_for_status()

        payload = r.json()

        if payload.get("s") != "ok":
            raise RuntimeError(f"Finnhub returned no data for {ticker}")

        df = pd.DataFrame({
            "date": pd.to_datetime(payload["t"], unit="s"),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"]
        })

        df["ticker"] = ticker

        return df
