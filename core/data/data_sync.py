"""
MarketSentinel — Data Sync Service

The ONLY module that calls Yahoo Finance (or TwelveData fallback)
at runtime. Everything else reads from PostgreSQL.

How it works:
  1. Load ticker list from MarketUniverse
  2. For each ticker, check the latest date stored in DB
  3. Fetch only missing dates from the market data provider
  4. Validate and store new rows in PostgreSQL
  5. Log results with full detail

Usage:
    from core.data.data_sync import DataSyncService

    # On app startup
    sync = DataSyncService()
    report = sync.sync_universe()

    # Single ticker
    report = sync.sync_ticker("AAPL")

Adding more stocks:
    Just add tickers to config/universe.json and restart.
    DataSync will detect new tickers (no DB rows) and fetch
    full history for them.
"""

import time
import threading
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.data.providers.market.router import MarketProviderRouter
from core.db.repository import OHLCVRepository
from core.logging.logger import get_logger

logger = get_logger(__name__)


class DataSyncService:
    """
    Synchronises market data from Yahoo Finance → PostgreSQL.

    This is the single point of contact with external market
    data APIs. After sync completes, all other services
    (inference, training, API endpoints) read from the DB.
    """

    # How many days of history to fetch for brand-new tickers
    DEFAULT_HISTORY_DAYS = 500

    # Minimum rows required for a valid fetch
    MIN_ROWS = 50

    # Pause between ticker fetches to respect rate limits
    BATCH_PAUSE_SEC = 0.3

    # How many calendar days ahead of DB latest to start fetching
    # (overlap ensures no gaps from weekends/holidays)
    OVERLAP_DAYS = 3

    def __init__(self) -> None:

        self._provider = MarketProviderRouter()

        self._lock = threading.Lock()

        logger.info(
            "DataSyncService initialized",
            extra={"component": "data_sync", "function": "__init__"},
        )

    # ─── Single ticker sync ──────────────────────────────

    def sync_ticker(
        self,
        ticker: str,
        force_full: bool = False,
    ) -> Dict:
        """
        Sync a single ticker from Yahoo → PostgreSQL.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            force_full: If True, ignore existing DB data and fetch full history

        Returns:
            Dict with sync result: {ticker, status, rows_fetched, rows_inserted, ...}
        """

        ticker = ticker.strip().upper()
        start_time = time.time()

        try:

            # ── Determine date range ─────────────────────

            end_date = pd.Timestamp.now(tz="UTC").normalize()

            if force_full:
                latest_in_db = None
            else:
                latest_in_db = OHLCVRepository.get_latest_date(ticker)

            if latest_in_db is None:
                # New ticker — fetch full history
                start_date = end_date - pd.Timedelta(days=self.DEFAULT_HISTORY_DAYS)
                sync_type = "full"
            else:
                # Existing ticker — fetch only missing days
                start_date = (
                    pd.Timestamp(latest_in_db, tz="UTC")
                    - pd.Timedelta(days=self.OVERLAP_DAYS)
                )
                sync_type = "delta"

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # ── Check if delta fetch is needed ───────────

            if sync_type == "delta":
                days_gap = (end_date - pd.Timestamp(latest_in_db, tz="UTC")).days
                if days_gap <= 1:
                    # DB is already up to date (today or yesterday)
                    return {
                        "ticker": ticker,
                        "status": "up_to_date",
                        "sync_type": "skip",
                        "rows_in_db": OHLCVRepository.get_row_count(ticker),
                        "latest_date": str(latest_in_db),
                        "duration_sec": round(time.time() - start_time, 2),
                    }

            # ── Fetch from provider ──────────────────────

            df = self._provider.fetch(
                ticker=ticker,
                start=start_str,
                end=end_str,
                interval="1d",
                min_rows=self.MIN_ROWS if sync_type == "full" else 1,
            )

            if df is None or df.empty:
                raise RuntimeError(f"Provider returned no data for {ticker}")

            rows_fetched = len(df)

            # ── Store in PostgreSQL ──────────────────────

            rows_inserted = OHLCVRepository.upsert_from_dataframe(df)

            duration = round(time.time() - start_time, 2)

            logger.info(
                "Sync complete | ticker=%s type=%s fetched=%d inserted=%d duration=%.2fs",
                ticker,
                sync_type,
                rows_fetched,
                rows_inserted,
                duration,
                extra={
                    "component": "data_sync",
                    "function": "sync_ticker",
                },
            )

            return {
                "ticker": ticker,
                "status": "ok",
                "sync_type": sync_type,
                "rows_fetched": rows_fetched,
                "rows_inserted": rows_inserted,
                "date_range": f"{start_str} → {end_str}",
                "duration_sec": duration,
            }

        except Exception as exc:

            duration = round(time.time() - start_time, 2)

            logger.warning(
                "Sync failed | ticker=%s error=%s duration=%.2fs",
                ticker,
                exc,
                duration,
                extra={
                    "component": "data_sync",
                    "function": "sync_ticker",
                },
            )

            return {
                "ticker": ticker,
                "status": "error",
                "error": str(exc),
                "duration_sec": duration,
            }

    # ─── Full universe sync ──────────────────────────────

    def sync_universe(
        self,
        tickers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Sync all tickers in the universe from Yahoo → PostgreSQL.

        Args:
            tickers: Optional list of tickers. If None, loads from MarketUniverse.

        Returns:
            Dict with overall sync report.
        """

        # Lazy import to avoid circular dependency at module level
        from core.market.universe import MarketUniverse

        start_time = time.time()

        if tickers is None:
            tickers = list(MarketUniverse.get_universe())

        logger.info(
            "Universe sync starting | tickers=%d",
            len(tickers),
            extra={"component": "data_sync", "function": "sync_universe"},
        )

        results: List[Dict] = []
        success_count = 0
        error_count = 0
        skip_count = 0
        total_inserted = 0

        for i, ticker in enumerate(tickers):

            result = self.sync_ticker(ticker)
            results.append(result)

            if result["status"] == "ok":
                success_count += 1
                total_inserted += result.get("rows_inserted", 0)
            elif result["status"] == "up_to_date":
                skip_count += 1
            else:
                error_count += 1

            # Rate limit pause between tickers
            if i < len(tickers) - 1:
                time.sleep(self.BATCH_PAUSE_SEC)

        total_duration = round(time.time() - start_time, 2)

        report = {
            "total_tickers": len(tickers),
            "synced": success_count,
            "skipped": skip_count,
            "errors": error_count,
            "total_rows_inserted": total_inserted,
            "duration_sec": total_duration,
            "details": results,
        }

        logger.info(
            "Universe sync complete | synced=%d skipped=%d errors=%d "
            "rows_inserted=%d duration=%.2fs",
            success_count,
            skip_count,
            error_count,
            total_inserted,
            total_duration,
            extra={"component": "data_sync", "function": "sync_universe"},
        )

        if error_count > 0:
            failed = [r["ticker"] for r in results if r["status"] == "error"]
            logger.warning(
                "Failed tickers: %s",
                failed,
                extra={"component": "data_sync", "function": "sync_universe"},
            )

        return report

    # ─── Health check ────────────────────────────────────

    def get_sync_status(self) -> Dict:
        """
        Quick status check: how many tickers have data in DB
        and what's the freshest date.
        """

        stored_tickers = OHLCVRepository.get_stored_tickers()

        latest_dates = {}
        for ticker in stored_tickers[:10]:  # sample first 10
            latest = OHLCVRepository.get_latest_date(ticker)
            if latest:
                latest_dates[ticker] = str(latest)

        return {
            "tickers_in_db": len(stored_tickers),
            "sample_latest_dates": latest_dates,
        }