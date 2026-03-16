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

    SOFT_FAIL_MODE = os.getenv("YAHOO_SOFT_FAIL", "1") == "1"
    SOFT_FAIL_RATIO = 0.70

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1.5

    RATE_LIMIT_WAIT = 4.0

    def __init__(self) -> None:

        self.fetcher = StockPriceFetcher()

        logger.debug("YahooProvider initialised.")

    @staticmethod
    def _normalize_datetime(series: pd.Series) -> pd.Series:

        dt = pd.to_datetime(series, errors="coerce", utc=True)

        if dt.isna().all():
            raise RuntimeError("Datetime parsing failed — all values are NaT.")

        return dt

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

    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame) -> pd.DataFrame:

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

    @staticmethod
    def _repair_ohlc(clean: pd.DataFrame) -> pd.DataFrame:
        """
        Repair common Yahoo OHLC inconsistencies.
        """

        high_fix = clean[["high", "open", "close"]].max(axis=1)
        low_fix = clean[["low", "open", "close"]].min(axis=1)

        clean["high"] = high_fix
        clean["low"] = low_fix

        return clean

    def _normalize(self, df: pd.DataFrame, ticker: str, min_rows: int) -> pd.DataFrame:

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError(f"Yahoo fetch returned empty DataFrame for {ticker}.")

        df = df.copy()

        df = self._flatten_columns(df)

        if "date" not in df.columns:

            if isinstance(df.index, pd.DatetimeIndex):

                idx = df.index

                idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

                df = df.reset_index(drop=True)

                df["date"] = idx

            else:

                raise RuntimeError(
                    f"Yahoo DataFrame for {ticker} has no datetime index or 'date' column."
                )

        clean = self._extract_ohlcv(df)

        clean["date"] = df["date"].values

        for col in ("open", "high", "low", "close", "volume"):

            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        clean.replace([np.inf, -np.inf], np.nan, inplace=True)

        clean.dropna(subset=["open", "high", "low", "close"], inplace=True)

        if clean.empty:
            raise RuntimeError(f"Normalization produced empty dataset for {ticker}.")

        clean["date"] = self._normalize_datetime(clean["date"])

        clean = (
            clean
            .drop_duplicates("date")
            .sort_values("date")
            .tail(self.MAX_ROWS)
            .reset_index(drop=True)
        )

        clean = self._repair_ohlc(clean)

        pct = clean["close"].pct_change().abs().fillna(0)

        extreme_mask = pct > self.MAX_DAILY_MOVE

        if extreme_mask.any():

            logger.warning(
                "Extreme price moves in Yahoo data for %s (%d bars)",
                ticker,
                extreme_mask.sum(),
            )

            clean.loc[extreme_mask, "close"] = np.nan

            clean["close"] = clean["close"].ffill().bfill()

        span_days = (clean["date"].max() - clean["date"].min()).days + 1

        if span_days > 0:

            density = clean["date"].nunique() / span_days

            if density < self.MIN_TRADING_DENSITY:

                logger.warning(
                    "Low trading density for %s (%.2f)",
                    ticker,
                    density,
                )

        clean["close"] = clean["close"].ffill().bfill()

        clean["volume"] = clean["volume"].fillna(0).clip(lower=0)

        if len(clean) < min_rows:

            soft_threshold = int(min_rows * self.SOFT_FAIL_RATIO)

            if self.SOFT_FAIL_MODE and len(clean) >= soft_threshold:

                logger.warning(
                    "Short history accepted in soft-fail mode for %s (%d rows).",
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

            try:

                raw_df = self.fetcher.fetch(
                    ticker,
                    start_date,
                    end_date,
                    yf_interval,
                )

                if raw_df is not None and not raw_df.empty:
                    break

            except Exception as exc:

                last_error = exc

                msg = str(exc).lower()

                if "too many requests" in msg or "rate limited" in msg or "429" in msg:

                    logger.warning(
                        "Yahoo rate limit detected for %s — cooling down.",
                        ticker,
                    )

                    time.sleep(self.RATE_LIMIT_WAIT)

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