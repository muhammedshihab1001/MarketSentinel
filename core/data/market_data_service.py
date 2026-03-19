"""
MarketSentinel — Market Data Service v5.1

Provides OHLCV price data to the inference pipeline, API endpoints,
and training pipeline. Reads EXCLUSIVELY from PostgreSQL.

Changes from v5.0:
  FIX: get_price_data_batch() now uses ThreadPoolExecutor(max_workers=8)
       for parallel DB reads.
       Sequential reads: 30 tickers × ~30ms = ~900ms
                         100 tickers × ~30ms = ~3000ms
       Parallel reads:   100 tickers / 8 workers = ~400ms
       SQLAlchemy get_session() is thread-safe (session-per-call).
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.db.repository import OHLCVRepository
from core.logging.logger import get_logger

logger = get_logger(__name__)

# Max parallel DB connections for batch reads
# Matches SQLAlchemy pool max_overflow default
BATCH_MAX_WORKERS = int(os.getenv("MARKET_MAX_WORKERS", "8"))


class MarketDataService:
    """
    Serves OHLCV data from PostgreSQL.

    Same public interface as before — get_price_data() and
    get_price_data_batch() — so all callers work without changes.
    """

    REQUIRED_COLUMNS = {
        "ticker", "date", "open", "high", "low", "close", "volume"
    }

    DEFAULT_MIN_HISTORY_ROWS = 60
    SAFE_LAG_DAYS = 2
    MAX_ROWS = 20_000
    MIN_TRADING_DENSITY = 0.55

    SOFT_FAIL_MODE = os.getenv("MARKET_SOFT_FAIL", "1") == "1"

    def __init__(self) -> None:

        logger.info(
            "MarketDataService initialized (DB-backed, parallel batch)",
            extra={"component": "market_data_service", "function": "__init__"},
        )

    # --------------------------------------------------
    # TICKER VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _sanitize_ticker(ticker: str) -> str:

        ticker = str(ticker).upper().strip()

        if not re.fullmatch(r"[A-Z0-9._-]{1,12}", ticker):
            raise RuntimeError(
                f"Unsafe or invalid ticker format: '{ticker}'"
            )

        return ticker

    # --------------------------------------------------
    # SAFE DATE CAP
    # --------------------------------------------------

    @classmethod
    def _cap_to_safe_date(cls, date_input: str) -> pd.Timestamp:

        requested = pd.Timestamp(date_input, tz="UTC").normalize()

        safe_cutoff = (
            pd.Timestamp.now(tz="UTC")
            .normalize()
            - pd.Timedelta(days=cls.SAFE_LAG_DAYS)
        )

        return min(requested, safe_cutoff)

    # --------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------

    def _validate_dataset(self, df: pd.DataFrame, ticker: str, min_rows: int):

        if df is None or df.empty:
            raise RuntimeError(f"Market data empty for {ticker}")

        df = df.copy()

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Schema violation for {ticker}. Missing: {missing}"
            )

        df["ticker"] = ticker

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["date"] = df["date"].dt.normalize()
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").drop_duplicates("date")

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        if not np.isfinite(df[numeric_cols].values).all():
            raise RuntimeError(f"Non-finite values detected for {ticker}")

        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise RuntimeError(f"Invalid price values detected for {ticker}")

        # OHLC sanity repair
        df["high"] = np.maximum.reduce([df["high"], df["open"], df["close"]])
        df["low"] = np.minimum.reduce([df["low"], df["open"], df["close"]])

        if df["close"].nunique() < 5:
            raise RuntimeError(f"Flat price series detected for {ticker}")

        df = df[df["volume"] >= 0]

        if len(df) > self.MAX_ROWS:
            df = df.tail(self.MAX_ROWS).reset_index(drop=True)

        span_days = (df["date"].max() - df["date"].min()).days + 1
        if span_days > 0:
            density = df["date"].nunique() / span_days
            if density < self.MIN_TRADING_DENSITY:
                logger.warning(
                    "Low trading density for %s: %.2f",
                    ticker, density,
                    extra={
                        "component": "market_data_service",
                        "function": "_validate_dataset",
                    },
                )

        if len(df) < min_rows:
            soft_threshold = int(min_rows * 0.7)
            if self.SOFT_FAIL_MODE and len(df) >= soft_threshold:
                logger.warning(
                    "Short history accepted for %s (%d rows)",
                    ticker, len(df),
                    extra={
                        "component": "market_data_service",
                        "function": "_validate_dataset",
                    },
                )
            else:
                raise RuntimeError(
                    f"Insufficient history for {ticker}: got {len(df)}"
                )

        return df.reset_index(drop=True)

    # --------------------------------------------------
    # SINGLE FETCH
    # --------------------------------------------------

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        min_history: Optional[int] = None,
    ) -> pd.DataFrame:

        ticker = self._sanitize_ticker(ticker)

        safe_end = self._cap_to_safe_date(end_date)
        end_str = safe_end.strftime("%Y-%m-%d")
        start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        if min_history is None:
            min_history = self.DEFAULT_MIN_HISTORY_ROWS

        start_time = time.time()

        df = OHLCVRepository.get_price_data(ticker, start_str, end_str)

        if df is None or df.empty:
            raise RuntimeError(
                f"No data in DB for {ticker} [{start_str} → {end_str}]. "
                f"Run data sync first."
            )

        df = self._validate_dataset(df, ticker, min_history)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        logger.info(
            "Market data served (DB) | ticker=%s rows=%d latency=%.1fms",
            ticker, len(df), latency_ms,
            extra={
                "component": "market_data_service",
                "function": "get_price_data",
            },
        )

        return df

    # --------------------------------------------------
    # BATCH FETCH — parallel with ThreadPoolExecutor
    # --------------------------------------------------

    def get_price_data_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        min_history: Optional[int] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Load OHLCV data for multiple tickers from PostgreSQL in parallel.

        Uses ThreadPoolExecutor(max_workers=8) for concurrent DB reads.
        SQLAlchemy get_session() is thread-safe — each call opens its own
        session from the connection pool.

        Performance: 100 tickers sequential = ~3000ms
                     100 tickers parallel (8 workers) = ~400ms
        """

        results: Dict[str, pd.DataFrame] = {}
        failures: Dict[str, str] = {}

        tickers = [
            self._sanitize_ticker(t)
            for t in list(dict.fromkeys(tickers))  # deduplicate, preserve order
        ]

        start_time = time.time()

        def _fetch_one(ticker: str):
            return ticker, self.get_price_data(
                ticker, start_date, end_date, interval, min_history
            )

        with ThreadPoolExecutor(max_workers=BATCH_MAX_WORKERS) as pool:

            futures = {pool.submit(_fetch_one, t): t for t in tickers}

            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    _, df = fut.result()
                    results[ticker] = df
                except Exception as exc:
                    failures[ticker] = str(exc)
                    logger.warning(
                        "Batch fetch failed | ticker=%s | error=%s",
                        ticker, exc,
                        extra={
                            "component": "market_data_service",
                            "function": "get_price_data_batch",
                        },
                    )

        if not results:
            raise RuntimeError(
                f"All {len(tickers)} tickers failed during batch fetch. "
                f"Is the database synced? Run DataSyncService.sync_universe() first."
            )

        if failures:
            logger.warning(
                "Batch partial failure | success=%d failed=%d",
                len(results), len(failures),
                extra={
                    "component": "market_data_service",
                    "function": "get_price_data_batch",
                },
            )

        total_ms = round((time.time() - start_time) * 1000, 1)

        logger.info(
            "Batch fetch complete (DB, parallel) | tickers=%d "
            "success=%d failed=%d total=%.1fms",
            len(tickers), len(results), len(failures), total_ms,
            extra={
                "component": "market_data_service",
                "function": "get_price_data_batch",
            },
        )

        return results, failures