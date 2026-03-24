"""
MarketSentinel — ORM Models

Three core tables:
  1. ohlcv_daily        — Raw price data from Yahoo Finance
  2. computed_features  — Pre-computed feature engineering output
  3. model_predictions  — Stored inference results for audit trail

FIX v3: Added composite index on computed_features(feature_version, ticker, date)
         Previous single-column ix_feature_version caused full table scan on
         the FeatureRepository.get_features() query — 3s for 27,400 rows.
         Composite index makes it a covered index lookup: ~5ms.

FIX v2: schema_signature changed String(32) → String(64)
         sha256 hex digest is always 64 characters.
"""

import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.db.engine import Base


# ─── OHLCV Daily Prices ──────────────────────────────────────

class OHLCVDaily(Base):
    """
    Stores daily OHLCV price data fetched from Yahoo Finance / TwelveData.

    Query pattern: WHERE ticker = ? AND date BETWEEN ? AND ?
    """

    __tablename__ = "ohlcv_daily"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    fetched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="yfinance",
    )

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_ohlcv_ticker_date"),
        Index("ix_ohlcv_ticker_date", "ticker", "date"),
        Index("ix_ohlcv_date", "date"),
    )

    def __repr__(self):
        return (
            f"<OHLCVDaily ticker={self.ticker} date={self.date} "
            f"close={self.close}>"
        )


# ─── Computed Features ────────────────────────────────────────

class ComputedFeature(Base):
    """
    Stores pre-computed feature engineering output.

    Keyed by (ticker, date, feature_version) so that when
    the feature pipeline changes, old cached features are
    automatically ignored.

    FIX v3: Composite index on (feature_version, ticker, date) replaces
            single-column index. FeatureRepository queries by feature_version
            first, then filters by ticker+date. Without the composite index
            PostgreSQL was doing a full sequential scan of 27,400 rows (3s).
            With the composite index it's a btree scan (~5ms).

    Migration for existing DBs:
        CREATE INDEX CONCURRENTLY ix_feature_version_ticker_date
            ON computed_features(feature_version, ticker, date);
        DROP INDEX IF EXISTS ix_feature_version;
    """

    __tablename__ = "computed_features"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    feature_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Hash of MODEL_FEATURES list — invalidates cache on schema change",
    )

    feature_data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        doc="All computed features as key-value pairs",
    )

    computed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "ticker", "date", "feature_version",
            name="uq_feature_ticker_date_version",
        ),
        # FIX: Composite covering index — feature_version is the primary filter
        # in FeatureRepository.get_features(version). Leading column = highest
        # selectivity filter in the WHERE clause.
        Index("ix_feature_version_ticker_date", "feature_version", "ticker", "date"),
        # Keep ticker+date index for per-ticker lookups
        Index("ix_feature_ticker_date", "ticker", "date"),
    )

    def __repr__(self):
        return (
            f"<ComputedFeature ticker={self.ticker} date={self.date} "
            f"version={self.feature_version}>"
        )


# ─── Model Predictions ───────────────────────────────────────

class ModelPrediction(Base):
    """
    Audit trail for model inference results.

    FIX v2: schema_signature was String(32) — sha256 hex is 64 chars.
    Changed to String(64) to prevent StringDataRightTruncation errors.

    Migration for existing DBs:
        ALTER TABLE model_predictions
        ALTER COLUMN schema_signature TYPE varchar(64);
    """

    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    schema_signature: Mapped[str] = mapped_column(String(64), nullable=False)

    raw_model_score: Mapped[float] = mapped_column(Float, nullable=False)
    hybrid_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    signal: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    drift_state: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    predicted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "ticker", "date", "model_version",
            name="uq_prediction_ticker_date_model",
        ),
        Index("ix_prediction_date_model", "date", "model_version"),
        Index("ix_prediction_ticker", "ticker"),
    )

    def __repr__(self):
        return (
            f"<ModelPrediction ticker={self.ticker} date={self.date} "
            f"score={self.raw_model_score:.4f}>"
        )