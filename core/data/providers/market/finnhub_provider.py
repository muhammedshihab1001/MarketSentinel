import os
import logging
import requests
import pandas as pd
import numpy as np

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.data.providers.market.base import MarketDataProvider


logger = logging.getLogger(__name__)


class FinnhubProvider(MarketDataProvider):

    URL = "https://finnhub.io/api/v1/stock/candle"

    REQUEST_TIMEOUT = (4, 15)

    MIN_ROWS = 120
    MAX_ROWS = 20_000
    FUTURE_TOLERANCE_SECONDS = 5

    REQUIRED_PAYLOAD_KEYS = {"t", "o", "h", "l", "c", "v"}

    INTERVAL_MAP = {
        "1d": "D",
        "1h": "60",
        "5m": "5"
    }

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
            allowed_methods=["GET"],
            raise_on_status=False,
            respect_retry_after_header=True
        )

        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=retry,
            pool_block=True
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

        try:
            ts = pd.Timestamp(dt)
        except Exception:
            raise RuntimeError(f"Invalid datetime supplied: {dt}")

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        return int(ts.timestamp())

    ########################################################

    def _validate_response(self, response):

        if response is None:
            raise RuntimeError("Finnhub returned no response.")

        if not response.content:
            raise RuntimeError("Finnhub returned empty body.")

        try:
            return response.json()
        except Exception:
            raise RuntimeError("Finnhub returned non-JSON response.")

    ########################################################

    def _validate_payload(self, payload, ticker):

        status = payload.get("s")

        if status == "no_data":
            raise RuntimeError(f"No market data available for {ticker}")

        if status != "ok":
            raise RuntimeError(f"Finnhub error for {ticker} | status={status}")

        missing = self.REQUIRED_PAYLOAD_KEYS - payload.keys()

        if missing:
            raise RuntimeError(
                f"Finnhub payload schema drift. Missing={missing}"
            )

        lengths = {len(payload[k]) for k in self.REQUIRED_PAYLOAD_KEYS}

        if len(lengths) != 1:
            raise RuntimeError("Finnhub payload arrays length mismatch.")

    ########################################################

    def _range_guard(self, df, start_date, end_date):

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        max_allowed = end_ts + pd.Timedelta(
            seconds=self.FUTURE_TOLERANCE_SECONDS
        )

        if df["date"].max() > max_allowed:
            raise RuntimeError(
                "Provider returned future candle — leakage risk."
            )

        if df["date"].min() < start_ts - pd.Timedelta(days=7):
            logger.warning(
                "Provider returned earlier-than-requested data."
            )

    ########################################################

    def _normalize_schema(self, payload, ticker, start_date, end_date):

        self._validate_payload(payload, ticker)

        df = pd.DataFrame({
            "date": pd.to_datetime(payload["t"], unit="s", utc=True),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"]
        })

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["date", "open", "high", "low", "close"])

        if df.empty:
            raise RuntimeError("Finnhub returned empty normalized dataset.")

        # price-only validation
        price_cols = ["open", "high", "low", "close"]

        if (df[price_cols] <= 0).any().any():
            raise RuntimeError("Non-positive price detected.")

        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            raise RuntimeError("High price invariant violated.")

        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            raise RuntimeError("Low price invariant violated.")

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        self._range_guard(df, start_date, end_date)

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Finnhub returned insufficient history.")

        df["ticker"] = ticker

        return df

    ########################################################

    def fetch(self, ticker, start_date, end_date, interval="1d"):

        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported interval for Finnhub: {interval}")

        resolution = self.INTERVAL_MAP[interval]

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

        payload = self._validate_response(response)

        df = self._normalize_schema(
            payload,
            ticker,
            start_date,
            end_date
        )

        logger.debug(
            "Finnhub served market data | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)
