import logging
import os
import time
import random
from typing import Optional

import numpy as np
import pandas as pd

from core.data.providers.market.base import MarketDataProvider
from core.data.data_fetcher import StockPriceFetcher

logger = logging.getLogger(__name__)


class YahooProvider(MarketDataProvider):

    PROVIDER_NAME = "yahoo"

    DEFAULT_MIN_ROWS = 120
    MAX_ROWS = 20_000

    ALLOWED_INTERVALS = {
        "1d", "D",
        "1wk",
        "1mo",
        "1h", "60m",
        "15m",
        "5m",
        "1m",
    }

    _INTERVAL_ALIAS = {
        "D": "1d",
        "60m": "1h",
    }

    MAX_DAILY_MOVE = 0.85
    MIN_TRADING_DENSITY = 0.50

    MAX_RETRIES = 2
    RETRY_DELAY_SECONDS = 1.5

    RATE_LIMIT_WAIT = 4.0
    FETCH_TIMEOUT_WARN = 10.0

    def __init__(self) -> None:

        self.fetcher = StockPriceFetcher()

        self.soft_fail_mode = os.getenv("YAHOO_SOFT_FAIL", "1") == "1"
        self.soft_fail_ratio = float(os.getenv("YAHOO_SOFT_FAIL_RATIO", "0.70"))

        logger.info("YahooProvider initialised (PRIMARY market data provider).")

    ############################################################
    # SAFE DATETIME NORMALIZATION (FIX TZ BUG)
    ############################################################

    @staticmethod
    def _normalize_datetime(series: pd.Series) -> pd.Series:
        """
        Convert any datetime series to normalised UTC dates safely.

        FIX 1: Uses pd.to_datetime(utc=True) to handle both tz-aware
        and tz-naive timestamps without the 'Cannot localize tz-aware
        Timestamp' crash.

        FIX 2: Changed isna().any() → isna().all(). A few NaT values
        from coercion are acceptable (dropped later). Only raise if
        ALL values are invalid.
        """

        dt = pd.to_datetime(series, utc=True, errors="coerce")

        if dt.isna().all():
            raise RuntimeError("Datetime parsing produced no valid timestamps.")

        dt = dt.dt.normalize()

        return dt

    ############################################################
    # COLUMN NORMALIZATION
    ############################################################

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = [
                "_".join(
                    str(lvl).strip().lower()
                    for lvl in col
                    if lvl is not None and str(lvl).strip() != ""
                )
                for col in df.columns
            ]

        else:

            df.columns = [str(c).strip().lower() for c in df.columns]

        df = df.loc[:, ~df.columns.duplicated()]

        return df

    ############################################################
    # OHLCV EXTRACTION
    ############################################################

    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame) -> pd.DataFrame:

        col_map = {}

        for col in df.columns:

            lc = col.lower()

            if "open" in lc and "open" not in col_map:
                col_map["open"] = col

            elif "high" in lc and "high" not in col_map:
                col_map["high"] = col

            elif "low" in lc and "low" not in col_map:
                col_map["low"] = col

            elif "adj" in lc and "close" in lc and "close" not in col_map:
                col_map["close"] = col

            elif "close" in lc and "close" not in col_map:
                col_map["close"] = col

            elif "volume" in lc and "volume" not in col_map:
                col_map["volume"] = col

        required = {"open", "high", "low", "close", "volume"}

        missing = required - set(col_map.keys())

        if missing:
            raise RuntimeError(f"Yahoo schema violation — missing columns: {missing}")

        clean = pd.DataFrame()

        for key in required:

            series = df[col_map[key]]

            if isinstance(series, pd.DataFrame):

                if series.shape[1] == 1:
                    series = series.iloc[:, 0]
                else:
                    raise RuntimeError(f"Ambiguous Yahoo column for '{key}'")

            clean[key] = series

        return clean

    ############################################################
    # OHLC REPAIR
    ############################################################

    @staticmethod
    def _repair_ohlc(clean: pd.DataFrame) -> pd.DataFrame:

        clean["high"] = clean[["high", "open", "close"]].max(axis=1)
        clean["low"] = clean[["low", "open", "close"]].min(axis=1)

        return clean

    ############################################################
    # NORMALIZATION PIPELINE
    ############################################################

    def _normalize(self, df: pd.DataFrame, ticker: str, min_rows: int) -> pd.DataFrame:

        ticker = ticker.strip().upper()

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError(f"Yahoo fetch returned empty DataFrame for {ticker}.")

        df = df.copy()

        df = self._flatten_columns(df)

        ############################################################
        # DATE EXTRACTION
        ############################################################

        if "date" not in df.columns:

            if isinstance(df.index, pd.DatetimeIndex):

                idx = pd.to_datetime(df.index, errors="coerce")

                df = df.reset_index(drop=True)

                df["date"] = idx

            else:

                raise RuntimeError(
                    f"Yahoo DataFrame for {ticker} has no datetime index or 'date' column."
                )

        ############################################################
        # EXTRACT OHLCV
        ############################################################

        clean = self._extract_ohlcv(df)

        clean["date"] = df["date"]

        ############################################################
        # NUMERIC SANITY
        ############################################################

        for col in ("open", "high", "low", "close", "volume"):
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        clean.replace([np.inf, -np.inf], np.nan, inplace=True)

        clean.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if clean.empty:
            raise RuntimeError(f"Normalization produced empty dataset for {ticker}.")

        ############################################################
        # FIX: DATE NORMALIZATION
        ############################################################

        clean["date"] = self._normalize_datetime(clean["date"])

        ############################################################
        # SORT + DEDUP
        ############################################################

        clean = (
            clean
            .drop_duplicates("date")
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        ############################################################
        # OHLC CONSISTENCY
        ############################################################

        clean = self._repair_ohlc(clean)

        ############################################################
        # OUTLIER PROTECTION (YFINANCE SPIKES)
        ############################################################

        pct = clean["close"].pct_change().abs().fillna(0)

        extreme_mask = pct > self.MAX_DAILY_MOVE

        if extreme_mask.any():

            logger.warning(
                "Extreme price moves detected | provider=yahoo ticker=%s bars=%d",
                ticker,
                extreme_mask.sum(),
            )

            clean.loc[extreme_mask, "close"] = np.nan
            clean["close"] = clean["close"].ffill().bfill()

        ############################################################
        # TRADING DENSITY CHECK
        ############################################################

        span_days = (clean["date"].max() - clean["date"].min()).days + 1

        if span_days > 0:

            density = clean["date"].nunique() / span_days

            if density < self.MIN_TRADING_DENSITY:

                logger.warning(
                    "Low trading density | provider=yahoo ticker=%s density=%.2f",
                    ticker,
                    density,
                )

        ############################################################
        # FINAL CLEAN
        ############################################################

        clean["close"] = clean["close"].ffill().bfill()
        clean["volume"] = clean["volume"].fillna(0).clip(lower=0)

        ############################################################
        # HISTORY CHECK
        ############################################################

        if len(clean) < min_rows:

            soft_threshold = int(min_rows * self.soft_fail_ratio)

            if self.soft_fail_mode and len(clean) >= soft_threshold:

                logger.warning(
                    "Short history accepted | provider=yahoo ticker=%s rows=%d",
                    ticker,
                    len(clean),
                )

            else:

                raise RuntimeError(
                    f"Insufficient history for {ticker}: got {len(clean)}, need {min_rows}"
                )

        clean["ticker"] = ticker

        logger.info(
            "Yahoo normalised | ticker=%s rows=%d",
            ticker,
            len(clean),
        )

        return clean

    ############################################################
    # FETCH
    ############################################################

    def fetch(self, ticker: str, start_date: str, end_date: str, interval: str, **kwargs) -> pd.DataFrame:

        if interval not in self.ALLOWED_INTERVALS:

            raise ValueError(
                f"YahooProvider: unsupported interval '{interval}'. "
                f"Allowed: {sorted(self.ALLOWED_INTERVALS)}"
            )

        yf_interval = self._INTERVAL_ALIAS.get(interval, interval)

        min_rows = kwargs.get("min_rows", self.DEFAULT_MIN_ROWS)

        raw_df: Optional[pd.DataFrame] = None
        last_error: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):

            start_time = time.time()

            try:

                raw_df = self.fetcher.fetch(
                    ticker,
                    start_date,
                    end_date,
                    yf_interval,
                )

                latency = time.time() - start_time

                if latency > self.FETCH_TIMEOUT_WARN:
                    logger.warning(
                        "Slow Yahoo fetch | ticker=%s latency=%.2fs",
                        ticker,
                        latency,
                    )

                if raw_df is not None and not raw_df.empty:
                    break

            except Exception as exc:

                last_error = exc

                msg = str(exc).lower()

                if any(x in msg for x in ["too many requests", "rate limit", "429"]):

                    logger.warning(
                        "Yahoo rate limit detected for %s — cooling down.",
                        ticker,
                    )

                    time.sleep(self.RATE_LIMIT_WAIT + random.uniform(0.5, 1.5))

                else:

                    logger.warning(
                        "Yahoo fetch error for %s (attempt %d/%d): %s",
                        ticker,
                        attempt,
                        self.MAX_RETRIES,
                        exc,
                    )

            if attempt < self.MAX_RETRIES:

                jitter = random.uniform(0.5, 1.5)

                time.sleep(self.RETRY_DELAY_SECONDS * jitter)

        if raw_df is None or raw_df.empty:

            msg = f"Yahoo returned no data for {ticker} after {self.MAX_RETRIES} attempts."

            if last_error:
                msg += f" Last error: {last_error}"

            raise RuntimeError(msg)

        normalised = self._normalize(
            df=raw_df,
            ticker=ticker,
            min_rows=min_rows,
        )

        return self.validate_contract(normalised)