"""
MarketSentinel — ORM Models

Three core tables:
  1. ohlcv_daily     — Raw price data from Yahoo Finance
  2. computed_features — Pre-computed feature engineering output
  3. model_predictions — Stored inference results for audit trail

All tables use (ticker, date) as the logical key with proper
indexes for the query patterns used in the inference pipeline.
"""

import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Float,
    BigInteger,
    DateTime,
    Date,
    Text,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB

from core.db.engine import Base


# ─── OHLCV Daily Prices ──────────────────────────────────────

class OHLCVDaily(Base):
    """
    Stores daily OHLCV price data fetched from Yahoo Finance.

    This is the primary cache layer that eliminates redundant
    API calls. On each request, only missing dates are fetched
    from Yahoo and appended here.

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

    # Metadata
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

    The feature_data column stores all computed features as
    a JSON blob — this avoids schema changes when new features
    are added to MODEL_FEATURES.

    Query pattern: WHERE ticker IN (?) AND date BETWEEN ? AND ?
                   AND feature_version = ?
    """

    __tablename__ = "computed_features"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    feature_version: Mapped[str] = mapped_column(
        String(32),
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
        Index("ix_feature_ticker_date", "ticker", "date"),
        Index("ix_feature_version", "feature_version"),
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

    Stores the raw model score and final hybrid score for
    each ticker on each prediction date. Useful for:
      - Debugging model behavior over time
      - Comparing predictions across model versions
      - Building a backtest from stored predictions

    Query pattern: WHERE date = ? AND model_version = ?
    """

    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_signature: Mapped[str] = mapped_column(String(32), nullable=False)

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