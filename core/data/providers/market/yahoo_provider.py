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
    # SAFE COLUMN FLATTENER
    ########################################################

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(
                    [str(level).strip().lower() for level in col if level]
                )
                for col in df.columns
            ]
        else:
            df.columns = [str(c).strip().lower() for c in df.columns]

        return df

    ########################################################
    # STRICT COLUMN EXTRACTION
    ########################################################

    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame, ticker: str):

        # Identify candidate columns
        col_map = {}

        for col in df.columns:

            if col.startswith("open"):
                col_map["open"] = col

            elif col.startswith("high"):
                col_map["high"] = col

            elif col.startswith("low"):
                col_map["low"] = col

            elif col.startswith("close"):
                col_map["close"] = col

            elif col.startswith("adj close"):
                col_map["close"] = col

            elif col.startswith("volume"):
                col_map["volume"] = col

        required = {"open", "high", "low", "close", "volume"}

        if not required.issubset(col_map.keys()):
            missing = required - set(col_map.keys())
            raise RuntimeError(f"Yahoo schema violation: {missing}")

        clean = pd.DataFrame({
            "open": df[col_map["open"]],
            "high": df[col_map["high"]],
            "low": df[col_map["low"]],
            "close": df[col_map["close"]],
            "volume": df[col_map["volume"]],
        })

        return clean

    ########################################################
    # CORE NORMALIZER
    ########################################################

    def _normalize(self, df, ticker, min_rows):

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("Yahoo fetch returned invalid dataframe.")

        df = df.copy()

        # Flatten columns FIRST
        df = self._flatten_columns(df)

        ####################################################
        # HANDLE DATE
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
        # EXTRACT STRICT OHLCV
        ####################################################

        clean = self._extract_ohlcv(df, ticker)

        clean["date"] = df["date"]

        ####################################################
        # NUMERIC HARDENING
        ####################################################

        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        clean.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if clean.empty:
            raise RuntimeError("Normalization produced empty dataset.")

        ####################################################
        # DATE NORMALIZATION
        ####################################################

        clean["date"] = self._normalize_datetime(clean["date"])

        ####################################################
        # SORT + DEDUPE
        ####################################################

        clean = (
            clean
            .drop_duplicates("date")
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        ####################################################
        # PRICE INVARIANT REPAIR
        ####################################################

        clean["high"] = clean[["high", "open", "close"]].max(axis=1)
        clean["low"] = clean[["low", "open", "close"]].min(axis=1)

        ####################################################
        # VOLUME SAFETY
        ####################################################

        clean["volume"] = clean["volume"].clip(lower=0).fillna(0)

        ####################################################
        # MIN HISTORY
        ####################################################

        if len(clean) < min_rows:
            raise RuntimeError(
                f"Insufficient history for {ticker} "
                f"({len(clean)} < {min_rows})"
            )

        clean["ticker"] = ticker

        logger.info(
            "Yahoo normalized | ticker=%s rows=%s",
            ticker,
            len(clean)
        )

        return clean

    ########################################################
    # PUBLIC FETCH
    ########################################################

    def fetch(
        self,
        ticker,
        start_date,
        end_date,
        interval,
        **kwargs
    ):

        if interval not in self.ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

        min_rows = kwargs.get("min_rows", self.DEFAULT_MIN_ROWS)

        raw_df = self.fetcher.fetch(
            ticker,
            start_date,
            end_date,
            interval
        )

        normalized = self._normalize(
            df=raw_df,
            ticker=ticker,
            min_rows=min_rows
        )

        return self.validate_contract(normalized)