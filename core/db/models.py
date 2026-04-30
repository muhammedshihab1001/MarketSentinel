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

FIX v4 (Issue #25): Added agent tracking and outcome fields to ModelPrediction
         - Individual agent outputs (signal, technical, political)
         - Actual forward returns for accuracy measurement
         - Prediction correctness flags for evaluation
         All new fields nullable for backward compatibility.
"""

import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
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
            f"<OHLCVDaily ticker={self.ticker} date={self.date} " f"close={self.close}>"
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
            "ticker",
            "date",
            "feature_version",
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
    Audit trail for model inference results with agent tracking.

    FIX v2: schema_signature was String(32) — sha256 hex is 64 chars.
    Changed to String(64) to prevent StringDataRightTruncation errors.

    FIX v4 (Issue #25): Added agent tracking and outcome fields.
    New fields enable:
      - Individual agent performance measurement
      - Prediction vs outcome comparison
      - Agent accuracy evaluation
      - Dynamic weight adjustment

    All new fields are nullable for backward compatibility with
    existing prediction records.

    Migration for existing DBs:
        See alembic/versions/001_add_agent_tracking_fields.py
    """

    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # ─── Core Prediction Fields (existing) ────────────────
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

    # ─── Signal Agent Outputs (Issue #25) ─────────────────
    signal_agent_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="SignalAgent consensus score (0.0-1.0)",
    )

    signal_agent_signal: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        doc="SignalAgent direction: LONG / SHORT / NEUTRAL",
    )

    signal_agent_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="SignalAgent confidence level (0.0-1.0)",
    )

    signal_agent_risk_level: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        doc="SignalAgent risk assessment: low / moderate / high / elevated",
    )

    # ─── Technical Agent Outputs (Issue #25) ──────────────
    technical_agent_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="TechnicalRiskAgent quality score (0.0-1.0)",
    )

    technical_agent_bias: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        doc="TechnicalRiskAgent market bias: bullish / bearish / neutral",
    )

    technical_agent_volatility_regime: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
        doc="Volatility regime: normal / high_volatility / low_volatility",
    )

    # ─── Political Agent Outputs (Issue #25) ──────────────
    political_agent_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="PoliticalRiskAgent geopolitical risk score (0.0-1.0)",
    )

    political_agent_label: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        doc="Political risk level: LOW / MEDIUM / HIGH / CRITICAL",
    )

    # ─── Outcome Tracking (Issue #25) ─────────────────────
    actual_forward_return: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Actual 5-day forward return (computed after outcome period)",
    )

    outcome_fetched_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when actual outcome was computed",
    )

    # ─── Evaluation Metrics (Issue #25) ───────────────────
    direction_correct: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        doc="True if predicted signal matched actual direction",
    )

    prediction_error: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Absolute error: |predicted_score - actual_return|",
    )

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "date",
            "model_version",
            name="uq_prediction_ticker_date_model",
        ),
        Index("ix_prediction_date_model", "date", "model_version"),
        Index("ix_prediction_ticker", "ticker"),
        # NEW: Index for outcome queries (Issue #25)
        Index(
            "ix_prediction_outcome_pending",
            "date",
            "outcome_fetched_at",
            postgresql_where="outcome_fetched_at IS NULL",
        ),
        # NEW: Index for agent evaluation queries (Issue #25)
        Index("ix_prediction_agent_eval", "date", "direction_correct"),
    )

    def __repr__(self):
        return (
            f"<ModelPrediction ticker={self.ticker} date={self.date} "
            f"score={self.raw_model_score:.4f}>"
        )