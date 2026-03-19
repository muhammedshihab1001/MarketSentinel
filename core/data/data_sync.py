"""
MarketSentinel — Data Sync Service v2.0

The ONLY module that calls Yahoo Finance (or TwelveData fallback)
at runtime. Everything else reads from PostgreSQL.

Changes from v1.0:
  - DEFAULT_HISTORY_DAYS: 500 → 730 (2 years)
    XGBoost needs 50,000+ samples. 100 tickers × 500 days = 50,000.
    Fetching 730 calendar days gets ~500 trading days reliably.
  - sync_universe() now uses ThreadPoolExecutor(max_workers=4)
    for parallel ticker syncs. Reduces full universe sync time
    from ~100 tickers × 0.3s = 30s → ~8s.
    Note: max_workers=4 not 8 — Yahoo Finance rate-limits
    concurrent requests. 4 is safe without triggering 429s.
  - Per-window history: training uses 730 days, inference/API
    use their own date windows from pipeline/routes directly.
    DataSync always syncs the full history window.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

from core.data.providers.market.router import MarketProviderRouter
from core.db.repository import OHLCVRepository
from core.logging.logger import get_logger

logger = get_logger(__name__)

# Max parallel Yahoo Finance fetches.
# Keep at 4 — Yahoo rate-limits concurrent requests.
# Higher values cause 429 errors.
SYNC_MAX_WORKERS = int(__import__("os").getenv("SYNC_MAX_WORKERS", "4"))


class DataSyncService:
    """
    Synchronises market data from Yahoo Finance → PostgreSQL.

    This is the single point of contact with external market
    data APIs. After sync completes, all other services
    (inference, training, API endpoints) read from the DB.
    """

    # 730 calendar days ≈ 500 trading days
    # Required for XGBoost to reach 50,000 training samples with 100 tickers
    DEFAULT_HISTORY_DAYS = 730

    # Minimum rows for a full fetch to be considered valid
    MIN_ROWS = 50

    # Overlap ensures no gaps from weekends/holidays at delta boundary
    OVERLAP_DAYS = 3

    def __init__(self) -> None:

        self._provider = MarketProviderRouter()
        self._lock = threading.Lock()

        logger.info(
            "DataSyncService initialized | history_days=%d | workers=%d",
            self.DEFAULT_HISTORY_DAYS,
            SYNC_MAX_WORKERS,
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
            ticker:     Stock ticker symbol (e.g. "AAPL")
            force_full: If True, ignore existing DB data and fetch full history

        Returns:
            Dict with sync result: {ticker, status, rows_fetched, rows_inserted, ...}
        """

        ticker = ticker.strip().upper()
        start_time = time.time()

        try:

            end_date = pd.Timestamp.now(tz="UTC").normalize()

            if force_full:
                latest_in_db = None
            else:
                latest_in_db = OHLCVRepository.get_latest_date(ticker)

            if latest_in_db is None:
                # New ticker — fetch full 2-year history
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

            # Skip if DB is already up to date
            if sync_type == "delta":
                days_gap = (end_date - pd.Timestamp(latest_in_db, tz="UTC")).days
                if days_gap <= 1:
                    return {
                        "ticker": ticker,
                        "status": "up_to_date",
                        "sync_type": "skip",
                        "rows_in_db": OHLCVRepository.get_row_count(ticker),
                        "latest_date": str(latest_in_db),
                        "duration_sec": round(time.time() - start_time, 2),
                    }

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
            rows_inserted = OHLCVRepository.upsert_from_dataframe(df)

            duration = round(time.time() - start_time, 2)

            logger.info(
                "Sync complete | ticker=%s type=%s fetched=%d inserted=%d duration=%.2fs",
                ticker, sync_type, rows_fetched, rows_inserted, duration,
                extra={"component": "data_sync", "function": "sync_ticker"},
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
                ticker, exc, duration,
                extra={"component": "data_sync", "function": "sync_ticker"},
            )

            return {
                "ticker": ticker,
                "status": "error",
                "error": str(exc),
                "duration_sec": duration,
            }

    # ─── Full universe sync — parallel ───────────────────

    def sync_universe(
        self,
        tickers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Sync all tickers in the universe from Yahoo → PostgreSQL.

        Uses ThreadPoolExecutor(max_workers=4) for parallel fetches.
        Yahoo Finance rate limit: safe at 4 concurrent, 429s at 8+.

        Args:
            tickers: Optional list. If None, loads from MarketUniverse.

        Returns:
            Dict with overall sync report.
        """

        from core.market.universe import MarketUniverse

        start_time = time.time()

        if tickers is None:
            tickers = list(MarketUniverse.get_universe())

        logger.info(
            "Universe sync starting | tickers=%d | workers=%d",
            len(tickers), SYNC_MAX_WORKERS,
            extra={"component": "data_sync", "function": "sync_universe"},
        )

        results: List[Dict] = []
        success_count = 0
        error_count = 0
        skip_count = 0
        total_inserted = 0

        # Parallel sync — respects Yahoo rate limits at max_workers=4
        with ThreadPoolExecutor(max_workers=SYNC_MAX_WORKERS) as pool:

            future_to_ticker = {
                pool.submit(self.sync_ticker, ticker): ticker
                for ticker in tickers
            }

            for fut in as_completed(future_to_ticker):
                ticker = future_to_ticker[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "ticker": ticker,
                        "status": "error",
                        "error": str(exc),
                        "duration_sec": 0,
                    }

                results.append(result)

                if result["status"] == "ok":
                    success_count += 1
                    total_inserted += result.get("rows_inserted", 0)
                elif result["status"] == "up_to_date":
                    skip_count += 1
                else:
                    error_count += 1

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
            success_count, skip_count, error_count,
            total_inserted, total_duration,
            extra={"component": "data_sync", "function": "sync_universe"},
        )

        if error_count > 0:
            failed = [r["ticker"] for r in results if r["status"] == "error"]
            logger.warning(
                "Failed tickers: %s", failed,
                extra={"component": "data_sync", "function": "sync_universe"},
            )

        return report

    # ─── Health check ────────────────────────────────────

    def get_sync_status(self) -> Dict:
        """Quick status: how many tickers in DB and freshest dates."""

        stored_tickers = OHLCVRepository.get_stored_tickers()

        latest_dates = {}
        for ticker in stored_tickers[:10]:
            latest = OHLCVRepository.get_latest_date(ticker)
            if latest:
                latest_dates[ticker] = str(latest)

        return {
            "tickers_in_db": len(stored_tickers),
            "sample_latest_dates": latest_dates,
        }