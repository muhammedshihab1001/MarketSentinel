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

    ALLOWED_INTERVALS = {
        "1d", "1wk", "1mo",
        "1m", "5m", "15m"
    }

    def __init__(self):
        self.fetcher = StockPriceFetcher()

    ########################################################
    # 🔥 PRODUCTION TZ NORMALIZER
    ########################################################

    def _normalize_datetime(self, series):

        dt = pd.to_datetime(series, errors="coerce")

        if dt.isna().all():
            raise RuntimeError("Datetime parsing failed.")

        # naive → localize
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")

        # aware → convert
        else:
            dt = dt.dt.tz_convert("UTC")

        return dt

    ########################################################

    def _validate_fetch_output(self, df):

        if df is None:
            raise RuntimeError("Yahoo fetcher returned None.")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("Yahoo fetcher returned non-DataFrame.")

        if df.empty:
            raise RuntimeError("Yahoo returned empty dataset.")

    ########################################################

    def _normalize(self, df, ticker, start_date, end_date):

        self._validate_fetch_output(df)

        df = df.copy()

        # HANDLE INDEX SAFELY
        if "date" not in df.columns:

            if isinstance(df.index, pd.DatetimeIndex):

                idx = df.index

                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                else:
                    idx = idx.tz_convert("UTC")

                df = df.reset_index(drop=True)
                df["date"] = idx

            else:
                raise RuntimeError("Yahoo index is not datetime.")

        df.columns = [c.lower().strip() for c in df.columns]

        # Adjusted close
        if "adj close" in df.columns:
            df["close"] = df["adj close"]

        required = {
            "date", "open", "high",
            "low", "close", "volume"
        }

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(f"Yahoo schema violation: {missing}")

        # numeric safety
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        if df.empty:
            raise RuntimeError("Normalization produced empty dataset.")

        # timezone fix
        df["date"] = self._normalize_datetime(df["date"])

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Yahoo returned insufficient history.")

        # price invariants
        if (df[["open","high","low","close"]] <= 0).any().any():
            raise RuntimeError("Non-positive price detected.")

        if (df["high"] < df[["open","close"]].max(axis=1)).any():
            raise RuntimeError("High invariant violated.")

        if (df["low"] > df[["open","close"]].min(axis=1)).any():
            raise RuntimeError("Low invariant violated.")

        df["ticker"] = ticker

        return df

    ########################################################

    def fetch(self, ticker, start_date, end_date, interval):

        if interval not in self.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

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
            "Yahoo served market data | %s rows=%s",
            ticker,
            len(df)
        )

        return self.validate_contract(df)
