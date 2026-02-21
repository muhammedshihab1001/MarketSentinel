import logging
import pandas as pd
import numpy as np

from core.data.providers.market.base import MarketDataProvider
from core.data.data_fetcher import StockPriceFetcher

logger = logging.getLogger(__name__)


class YahooProvider(MarketDataProvider):

    PROVIDER_NAME = "yahoo"

    DEFAULT_MIN_ROWS = 120
    MAX_ROWS = 20_000

    ALLOWED_INTERVALS = {
        "1d", "1wk", "1mo",
        "1m", "5m", "15m"
    }

    ########################################################

    def __init__(self):
        self.fetcher = StockPriceFetcher()

    ########################################################
    # SAFE DATETIME NORMALIZATION
    ########################################################

    @staticmethod
    def _normalize_datetime(series):

        dt = pd.to_datetime(series, errors="coerce")

        if dt.isna().all():
            raise RuntimeError("Datetime parsing failed.")

        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        else:
            dt = dt.dt.tz_convert("UTC")

        return dt

    ########################################################
    # FETCH OUTPUT VALIDATION
    ########################################################

    @staticmethod
    def _validate_fetch_output(df):

        if df is None:
            raise RuntimeError("Yahoo fetcher returned None.")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("Yahoo fetcher returned non-DataFrame.")

        if df.empty:
            raise RuntimeError("Yahoo returned empty dataset.")

    ########################################################
    # CORE NORMALIZER
    ########################################################

    def _normalize(
        self,
        df,
        ticker,
        min_rows
    ):

        self._validate_fetch_output(df)

        df = df.copy()

        ####################################################
        # HANDLE INDEX → date column
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

        ####################################################
        # COLUMN STANDARDIZATION
        ####################################################

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

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if df.empty:
            raise RuntimeError("Normalization produced empty dataset.")

        ####################################################
        # DATE NORMALIZATION
        ####################################################

        df["date"] = self._normalize_datetime(df["date"])

        ####################################################
        # SORT + DEDUPE + ROW CAP
        ####################################################

        df = (
            df
            .drop_duplicates("date")
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        ####################################################
        # PRICE SAFETY FILTER
        ####################################################

        df = df[
            (df["open"] > 0) &
            (df["high"] > 0) &
            (df["low"] > 0) &
            (df["close"] > 0)
        ]

        if df.empty:
            raise RuntimeError("All rows invalid after price filter.")

        # Repair high/low invariants
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        ####################################################
        # VOLUME SAFETY
        ####################################################

        df["volume"] = df["volume"].clip(lower=0)
        df["volume"] = df["volume"].fillna(0)

        ####################################################
        # CONFIGURABLE MIN HISTORY
        ####################################################

        if len(df) < min_rows:
            raise RuntimeError(
                f"Insufficient history for {ticker} "
                f"({len(df)} < {min_rows})"
            )

        ####################################################
        # FINALIZE
        ####################################################

        df["ticker"] = ticker

        logger.info(
            "Yahoo normalized | ticker=%s rows=%s",
            ticker,
            len(df)
        )

        return df

    ########################################################
    # PUBLIC FETCH
    ########################################################

    def fetch(
        self,
        ticker,
        start_date,
        end_date,
        interval,
        min_rows=None
    ):

        if interval not in self.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

        if min_rows is None:
            min_rows = self.DEFAULT_MIN_ROWS

        df = self.fetcher.fetch(
            ticker,
            start_date,
            end_date,
            interval
        )

        df = self._normalize(
            df=df,
            ticker=ticker,
            min_rows=min_rows
        )

        return self.validate_contract(df)