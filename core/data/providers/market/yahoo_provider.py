import logging
import pandas as pd
import numpy as np

from core.data.providers.market.base import MarketDataProvider
from core.data.data_fetcher import StockPriceFetcher


logger = logging.getLogger(__name__)


class YahooProvider(MarketDataProvider):

    PROVIDER_NAME = "yahoo"

    MIN_ROWS = 120
    MAX_ROWS = 20_000
    FUTURE_TOLERANCE_SECONDS = 5

    ALLOWED_INTERVALS = {
        "1d",
        "1wk",
        "1mo",
        "1m",
        "5m",
        "15m"
    }

    def __init__(self):
        self.fetcher = StockPriceFetcher()

    ########################################################

    def _validate_fetch_output(self, df):

        if df is None:
            raise RuntimeError("Yahoo fetcher returned None.")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("Yahoo fetcher returned non-DataFrame.")

        if df.empty:
            raise RuntimeError("Yahoo returned empty dataset.")

    ########################################################

    def _range_guard(self, df, start_date, end_date):

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        max_allowed = end_ts + pd.Timedelta(
            seconds=self.FUTURE_TOLERANCE_SECONDS
        )

        if df["date"].max() > max_allowed:
            raise RuntimeError(
                "Yahoo returned future candle — leakage risk."
            )

        if df["date"].min() < start_ts - pd.Timedelta(days=7):
            logger.warning(
                "Yahoo returned earlier-than-requested data."
            )

    ########################################################

    def _normalize(self, df: pd.DataFrame, ticker: str,
                   start_date, end_date):

        self._validate_fetch_output(df)

        df = df.copy()

        # Handle index-based datetime
        if "date" not in df.columns:

            idx_name = (df.index.name or "").lower()

            if "date" in idx_name or "time" in idx_name:
                df = df.reset_index()
                df.rename(columns={df.columns[0]: "date"}, inplace=True)
            else:
                raise RuntimeError("Yahoo index does not contain datetime.")

        df.columns = [c.lower().strip() for c in df.columns]

        # Explicit adjusted close handling
        if "adj close" in df.columns:
            logger.debug("Using adjusted close for %s", ticker)
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

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["date", "open", "high", "low", "close"])

        if df.empty:
            raise RuntimeError("Yahoo normalization produced empty dataset.")

        # price-only validation
        price_cols = ["open", "high", "low", "close"]

        if (df[price_cols] <= 0).any().any():
            raise RuntimeError("Non-positive price detected.")

        if (df["volume"] < 0).any():
            raise RuntimeError("Negative volume detected.")

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
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        self._range_guard(df, start_date, end_date)

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Yahoo returned insufficient history.")

        df["ticker"] = ticker

        return df

    ########################################################

    def fetch(self, ticker, start_date, end_date, interval):

        if interval not in self.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval for Yahoo: {interval}")

        df = self.fetcher.fetch(
            ticker,
            start_date,
            end_date,
            interval
        )

        df = self._normalize(
            df,
            ticker,
            start_date,
            end_date
        )

        logger.debug(
            "Yahoo served market data | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)
