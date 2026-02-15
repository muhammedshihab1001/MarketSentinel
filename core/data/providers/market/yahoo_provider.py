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
    # SAFE DATETIME
    ########################################################

    def _normalize_datetime(self, series):

        dt = pd.to_datetime(series, errors="coerce")

        if dt.isna().all():
            raise RuntimeError("Datetime parsing failed.")

        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
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
    # 🔥 INSTITUTIONAL NORMALIZER
    ########################################################

    def _normalize(self, df, ticker, start_date, end_date):

        self._validate_fetch_output(df)

        df = df.copy()

        ####################################################
        # HANDLE INDEX
        ####################################################

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

        if "adj close" in df.columns:
            df["close"] = df["adj close"]

        required = {
            "date", "open", "high",
            "low", "close", "volume"
        }

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(f"Yahoo schema violation: {missing}")

        ####################################################
        # NUMERIC HARDENING
        ####################################################

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with missing price — SAFE
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if df.empty:
            raise RuntimeError("Normalization produced empty dataset.")

        ####################################################
        # TIMEZONE
        ####################################################

        df["date"] = self._normalize_datetime(df["date"])

        ####################################################
        # SORT + DEDUPE
        ####################################################

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Yahoo returned insufficient history.")

        ####################################################
        # 🔥 SOFT PRICE REPAIR (CRITICAL)
        ####################################################

        # Remove non-positive prices safely
        df = df[
            (df["open"] > 0) &
            (df["high"] > 0) &
            (df["low"] > 0) &
            (df["close"] > 0)
        ]

        if df.empty:
            raise RuntimeError("All rows invalid after price filter.")

        # Repair high / low anomalies instead of crashing
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        ####################################################
        # VOLUME REPAIR
        ####################################################

        df["volume"] = df["volume"].clip(lower=0)
        df["volume"] = df["volume"].fillna(0)

        ####################################################
        # FINAL SAFETY
        ####################################################

        if len(df) < self.MIN_ROWS:
            raise RuntimeError("Dataset collapsed after repair.")

        df["ticker"] = ticker

        logger.info(
            "Yahoo normalized | ticker=%s rows=%s",
            ticker,
            len(df)
        )

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

        return self.validate_contract(df)
