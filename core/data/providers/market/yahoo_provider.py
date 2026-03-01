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
    def _normalize_datetime(series: pd.Series):

        if not isinstance(series, pd.Series):
            raise RuntimeError("Date column must be a pandas Series.")

        dt = pd.to_datetime(series, errors="coerce", utc=True)

        if dt.isna().all():
            raise RuntimeError("Datetime parsing failed.")

        return dt

    ########################################################
    # SAFE COLUMN FLATTENER
    ########################################################

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(
                    str(level).strip().lower()
                    for level in col
                    if level is not None and str(level).strip() != ""
                )
                for col in df.columns
            ]
        else:
            df.columns = [str(c).strip().lower() for c in df.columns]

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    ########################################################
    # STRICT COLUMN EXTRACTION
    ########################################################

    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame, ticker: str):

        col_map = {}

        for col in df.columns:

            lc = col.lower()

            if lc.startswith("open") and "open" not in col_map:
                col_map["open"] = col

            elif lc.startswith("high") and "high" not in col_map:
                col_map["high"] = col

            elif lc.startswith("low") and "low" not in col_map:
                col_map["low"] = col

            elif lc.startswith("adj close") and "close" not in col_map:
                col_map["close"] = col

            elif lc.startswith("close") and "close" not in col_map:
                col_map["close"] = col

            elif lc.startswith("volume") and "volume" not in col_map:
                col_map["volume"] = col

        required = {"open", "high", "low", "close", "volume"}

        if not required.issubset(col_map.keys()):
            missing = required - set(col_map.keys())
            raise RuntimeError(f"Yahoo schema violation: {missing}")

        clean = pd.DataFrame()

        for key in required:

            series = df[col_map[key]]

            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 1:
                    series = series.iloc[:, 0]
                else:
                    raise RuntimeError(
                        f"Ambiguous Yahoo column for {key}"
                    )

            clean[key] = series

        return clean

    ########################################################
    # CORE NORMALIZER
    ########################################################

    def _normalize(self, df, ticker, min_rows):

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("Yahoo fetch returned empty dataframe.")

        df = df.copy()
        df = self._flatten_columns(df)

        ####################################################
        # HANDLE DATE COLUMN
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
        # PRICE REPAIR
        ####################################################

        clean["high"] = clean[["high", "open", "close"]].max(axis=1)
        clean["low"] = clean[["low", "open", "close"]].min(axis=1)

        # Remove zero or negative prices
        clean = clean[clean["close"] > 0]

        ####################################################
        # SMALL GAP REPAIR
        ####################################################

        clean["close"] = clean["close"].ffill().bfill()

        ####################################################
        # VOLUME SAFETY
        ####################################################

        clean["volume"] = clean["volume"].fillna(0).clip(lower=0)

        ####################################################
        # INTRADAY ADJUSTMENT
        ####################################################

        if min_rows < 60:
            min_rows = max(min_rows, 30)

        ####################################################
        # MIN HISTORY CHECK
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