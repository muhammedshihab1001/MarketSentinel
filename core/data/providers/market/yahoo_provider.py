import logging
import pandas as pd
import numpy as np

from core.data.providers.market.base import MarketDataProvider
from core.data.data_fetcher import StockPriceFetcher


logger = logging.getLogger(__name__)


class YahooProvider(MarketDataProvider):

    PROVIDER_NAME = "yahoo"

    MIN_ROWS = 120

    ALLOWED_INTERVALS = {
        "1d",
        "1wk",
        "1mo",
        "1m",
        "5m",
        "15m"
    }

    REQUIRED_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ticker"
    }

    def __init__(self):
        self.fetcher = StockPriceFetcher()

    def _normalize(self, df: pd.DataFrame, ticker: str):

        if df is None or df.empty:
            raise RuntimeError("Yahoo returned empty dataset.")

        df = df.copy()

        if "date" not in df.columns:

            idx_name = (df.index.name or "").lower()

            if "date" in idx_name or "time" in idx_name:
                df = df.reset_index()
                df.rename(columns={df.columns[0]: "date"}, inplace=True)
            else:
                raise RuntimeError("Yahoo index does not contain datetime.")

        df.columns = [c.lower().strip() for c in df.columns]

        if "adj close" in df.columns:
            df["close"] = df["adj close"]

        required_price_cols = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        }

        missing = required_price_cols - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Yahoo schema violation. Missing={missing}"
            )

        df["ticker"] = ticker

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=numeric_cols)

        if df.empty:
            raise RuntimeError("Yahoo normalization produced empty dataset.")

        if (df["volume"] < 0).any():
            raise RuntimeError("Negative volume detected.")

        if (df[numeric_cols[:-1]] <= 0).any().any():
            raise RuntimeError("Non-positive price detected.")

        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            raise RuntimeError("High invariant violated.")

        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            raise RuntimeError("Low invariant violated.")

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce"
        )

        df = df.dropna(subset=["date"])

        df = (
            df
            .drop_duplicates(subset=["ticker", "date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Yahoo returned insufficient history.")

        return df

    def fetch(self, ticker, start_date, end_date, interval):

        if interval not in self.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval for Yahoo: {interval}")

        df = self.fetcher.fetch(
            ticker,
            start_date,
            end_date,
            interval
        )

        df = self._normalize(df, ticker)

        logger.debug(
            "Yahoo served market data | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)
