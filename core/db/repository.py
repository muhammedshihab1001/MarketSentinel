"""
MarketSentinel — Database Repository

CRUD operations for all three tables. Designed for the specific
query patterns used in the inference pipeline:

  - OHLCVRepository: bulk upsert prices, get date ranges, find gaps
  - FeatureRepository: store/load feature blobs, invalidate on version change
  - PredictionRepository: store predictions, query by date/model
"""

import datetime
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import select, delete, func, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.db.engine import get_session
from core.db.models import OHLCVDaily, ComputedFeature, ModelPrediction
from core.logging.logger import get_logger

logger = get_logger(__name__)


# ─── OHLCV Repository ────────────────────────────────────────

class OHLCVRepository:
    """
    Handles all database operations for daily price data.

    Key method: upsert_from_dataframe() — takes a pandas DataFrame
    from StockPriceFetcher and writes it to the DB, skipping
    rows that already exist (ON CONFLICT DO NOTHING).
    """

    @staticmethod
    def get_price_data(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data from DB for a single ticker + date range.

        Returns None if no data found.
        """

        with get_session() as session:

            stmt = (
                select(OHLCVDaily)
                .where(
                    OHLCVDaily.ticker == ticker.upper(),
                    OHLCVDaily.date >= start_date,
                    OHLCVDaily.date <= end_date,
                )
                .order_by(OHLCVDaily.date)
            )

            rows = session.execute(stmt).scalars().all()

            if not rows:
                return None

            records = [
                {
                    "ticker": r.ticker,
                    "date": pd.Timestamp(r.date, tz="UTC"),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]

            df = pd.DataFrame(records)

            logger.debug(
                "DB price load | ticker=%s rows=%d",
                ticker,
                len(df),
                extra={"component": "db.repository", "function": "get_price_data"},
            )

            return df

    @staticmethod
    def get_latest_date(ticker: str) -> Optional[datetime.date]:
        """Return the most recent date stored for a ticker, or None."""

        with get_session() as session:

            stmt = (
                select(func.max(OHLCVDaily.date))
                .where(OHLCVDaily.ticker == ticker.upper())
            )

            result = session.execute(stmt).scalar()

            return result

    @staticmethod
    def upsert_from_dataframe(df: pd.DataFrame) -> int:
        """
        Bulk upsert OHLCV rows from a pandas DataFrame.

        Uses PostgreSQL ON CONFLICT DO NOTHING to skip duplicates.
        Returns the number of rows inserted.
        """

        if df is None or df.empty:
            return 0

        records = []

        for _, row in df.iterrows():

            date_val = pd.Timestamp(row["date"]).date()

            records.append({
                "ticker": str(row["ticker"]).upper(),
                "date": date_val,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "source": "yfinance",
            })

        if not records:
            return 0

        with get_session() as session:

            stmt = pg_insert(OHLCVDaily).values(records)

            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_ohlcv_ticker_date"
            )

            result = session.execute(stmt)

            inserted = result.rowcount if result.rowcount else 0

            logger.info(
                "DB price upsert | ticker=%s total=%d inserted=%d",
                records[0]["ticker"] if records else "?",
                len(records),
                inserted,
                extra={
                    "component": "db.repository",
                    "function": "upsert_from_dataframe",
                },
            )

            return inserted

    @staticmethod
    def get_stored_tickers() -> List[str]:
        """Return list of all tickers that have stored price data."""

        with get_session() as session:

            stmt = select(OHLCVDaily.ticker).distinct()
            rows = session.execute(stmt).scalars().all()

            return sorted(rows)

    @staticmethod
    def get_row_count(ticker: str) -> int:
        """Return total number of stored rows for a ticker."""

        with get_session() as session:

            stmt = (
                select(func.count(OHLCVDaily.id))
                .where(OHLCVDaily.ticker == ticker.upper())
            )

            return session.execute(stmt).scalar() or 0


# ─── Feature Repository ──────────────────────────────────────

class FeatureRepository:
    """
    Handles storage and retrieval of pre-computed features.

    Features are stored as JSONB blobs keyed by
    (ticker, date, feature_version). When MODEL_FEATURES
    changes, the version hash changes and old cache is
    automatically bypassed.
    """

    @staticmethod
    def get_features(
        tickers: List[str],
        start_date: str,
        end_date: str,
        feature_version: str,
    ) -> Optional[pd.DataFrame]:
        """
        Load cached features for multiple tickers.

        Returns None if no cached features found.
        """

        with get_session() as session:

            stmt = (
                select(ComputedFeature)
                .where(
                    ComputedFeature.ticker.in_([t.upper() for t in tickers]),
                    ComputedFeature.date >= start_date,
                    ComputedFeature.date <= end_date,
                    ComputedFeature.feature_version == feature_version,
                )
                .order_by(ComputedFeature.ticker, ComputedFeature.date)
            )

            rows = session.execute(stmt).scalars().all()

            if not rows:
                return None

            records = []

            for r in rows:
                record = {
                    "ticker": r.ticker,
                    "date": pd.Timestamp(r.date, tz="UTC"),
                    **r.feature_data,
                }
                records.append(record)

            df = pd.DataFrame(records)

            logger.debug(
                "DB feature load | tickers=%d rows=%d version=%s",
                len(tickers),
                len(df),
                feature_version[:8],
                extra={"component": "db.repository", "function": "get_features"},
            )

            return df

    @staticmethod
    def store_features(
        df: pd.DataFrame,
        feature_version: str,
        feature_columns: List[str],
    ) -> int:
        """
        Bulk store computed features from a DataFrame.

        Extracts only the specified feature_columns into the
        JSONB blob, plus ticker and date as keys.
        """

        if df is None or df.empty:
            return 0

        records = []

        available_cols = [c for c in feature_columns if c in df.columns]

        for _, row in df.iterrows():

            feature_data = {
                col: float(row[col]) if pd.notna(row[col]) else 0.0
                for col in available_cols
            }

            records.append({
                "ticker": str(row["ticker"]).upper(),
                "date": pd.Timestamp(row["date"]).date(),
                "feature_version": feature_version,
                "feature_data": feature_data,
            })

        if not records:
            return 0

        with get_session() as session:

            stmt = pg_insert(ComputedFeature).values(records)

            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_feature_ticker_date_version"
            )

            result = session.execute(stmt)
            inserted = result.rowcount if result.rowcount else 0

            logger.info(
                "DB feature store | rows=%d inserted=%d version=%s",
                len(records),
                inserted,
                feature_version[:8],
                extra={
                    "component": "db.repository",
                    "function": "store_features",
                },
            )

            return inserted

    @staticmethod
    def invalidate_version(feature_version: str) -> int:
        """Delete all cached features for a specific version."""

        with get_session() as session:

            stmt = (
                delete(ComputedFeature)
                .where(ComputedFeature.feature_version == feature_version)
            )

            result = session.execute(stmt)

            deleted = result.rowcount if result.rowcount else 0

            logger.info(
                "DB feature invalidation | version=%s deleted=%d",
                feature_version[:8],
                deleted,
                extra={
                    "component": "db.repository",
                    "function": "invalidate_version",
                },
            )

            return deleted


# ─── Prediction Repository ────────────────────────────────────

class PredictionRepository:
    """
    Stores model prediction results for audit and backtesting.
    """

    @staticmethod
    def store_predictions(predictions: List[dict]) -> int:
        """
        Bulk store prediction results.

        Each dict should have: ticker, date, model_version,
        schema_signature, raw_model_score, hybrid_score,
        weight, signal, drift_state.
        """

        if not predictions:
            return 0

        records = []

        for pred in predictions:

            records.append({
                "ticker": str(pred["ticker"]).upper(),
                "date": pd.Timestamp(pred["date"]).date(),
                "model_version": pred.get("model_version", "unknown"),
                "schema_signature": pred.get("schema_signature", "unknown"),
                "raw_model_score": float(pred.get("raw_model_score", 0)),
                "hybrid_score": float(pred.get("hybrid_score", 0)),
                "weight": float(pred.get("weight", 0)),
                "signal": pred.get("signal"),
                "drift_state": pred.get("drift_state"),
            })

        with get_session() as session:

            stmt = pg_insert(ModelPrediction).values(records)

            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_prediction_ticker_date_model"
            )

            result = session.execute(stmt)
            inserted = result.rowcount if result.rowcount else 0

            logger.info(
                "DB prediction store | total=%d inserted=%d",
                len(records),
                inserted,
                extra={
                    "component": "db.repository",
                    "function": "store_predictions",
                },
            )

            return inserted

    @staticmethod
    def get_predictions(
        date: str,
        model_version: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Load predictions for a specific date."""

        with get_session() as session:

            stmt = (
                select(ModelPrediction)
                .where(ModelPrediction.date == date)
            )

            if model_version:
                stmt = stmt.where(
                    ModelPrediction.model_version == model_version
                )

            stmt = stmt.order_by(ModelPrediction.raw_model_score.desc())

            rows = session.execute(stmt).scalars().all()

            if not rows:
                return None

            records = [
                {
                    "ticker": r.ticker,
                    "date": str(r.date),
                    "model_version": r.model_version,
                    "raw_model_score": r.raw_model_score,
                    "hybrid_score": r.hybrid_score,
                    "weight": r.weight,
                    "signal": r.signal,
                    "drift_state": r.drift_state,
                    "predicted_at": str(r.predicted_at),
                }
                for r in rows
            ]

            return pd.DataFrame(records)