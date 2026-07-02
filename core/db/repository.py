"""
MarketSentinel — Database Repository v2.4

Changes from v2.3:
  NEW (Issue #27): Added agent performance methods:
        - get_predictions_with_outcomes() — Query predictions for agent eval
        - store_agent_performance() — Batch store agent metrics

  NEW (Issue #26): Added outcome computation methods:
        - get_predictions_needing_outcomes() — Query pending outcomes
        - update_prediction_outcomes() — Batch update outcomes

  FIX (Issue #25): Extended store_predictions() to save agent outputs.

  PERF: All methods use vectorised pandas operations.

  NO CHANGE to public API — all callers work without modification.
"""

import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, delete, func, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.db.engine import get_session
from core.db.models import OHLCVDaily, ComputedFeature, ModelPrediction, AgentPerformance
from core.logging.logger import get_logger

logger = get_logger(__name__)


# ─── OHLCV Repository ────────────────────────────────────────


class OHLCVRepository:
    """
    Handles all database operations for daily price data.
    """

    @staticmethod
    def get_price_data(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:

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

        with get_session() as session:

            stmt = select(func.max(OHLCVDaily.date)).where(
                OHLCVDaily.ticker == ticker.upper()
            )

            return session.execute(stmt).scalar()

    @staticmethod
    def upsert_from_dataframe(df: pd.DataFrame) -> int:

        if df is None or df.empty:
            return 0

        work = df.copy()
        work["ticker"] = work["ticker"].str.upper()
        work["date"] = pd.to_datetime(work["date"]).dt.date
        work["open"] = pd.to_numeric(work["open"], errors="coerce")
        work["high"] = pd.to_numeric(work["high"], errors="coerce")
        work["low"] = pd.to_numeric(work["low"], errors="coerce")
        work["close"] = pd.to_numeric(work["close"], errors="coerce")
        work["volume"] = pd.to_numeric(work.get("volume", 0), errors="coerce").fillna(0)
        work["source"] = "yfinance"

        records = work[
            ["ticker", "date", "open", "high", "low", "close", "volume", "source"]
        ].to_dict("records")

        if not records:
            return 0

        with get_session() as session:

            stmt = pg_insert(OHLCVDaily).values(records)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_ohlcv_ticker_date")
            result = session.execute(stmt)
            inserted = result.rowcount if result.rowcount else 0

            logger.info(
                "DB price upsert | ticker=%s total=%d inserted=%d",
                records[0]["ticker"],
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

        with get_session() as session:
            stmt = select(OHLCVDaily.ticker).distinct()
            return sorted(session.execute(stmt).scalars().all())

    @staticmethod
    def get_row_count(ticker: str) -> int:

        with get_session() as session:
            stmt = select(func.count(OHLCVDaily.id)).where(
                OHLCVDaily.ticker == ticker.upper()
            )
            return session.execute(stmt).scalar() or 0


# ─── Feature Repository ──────────────────────────────────────


class FeatureRepository:
    """
    Storage and retrieval of pre-computed features.
    """

    @staticmethod
    def get_features(
        tickers: List[str],
        start_date: str,
        end_date: str,
        feature_version: str,
    ) -> Optional[pd.DataFrame]:

        with get_session() as session:

            stmt = (
                select(
                    ComputedFeature.ticker,
                    ComputedFeature.date,
                    ComputedFeature.feature_data,
                )
                .where(
                    ComputedFeature.ticker.in_([t.upper() for t in tickers]),
                    ComputedFeature.date >= start_date,
                    ComputedFeature.date <= end_date,
                    ComputedFeature.feature_version == feature_version,
                )
                .order_by(ComputedFeature.ticker, ComputedFeature.date)
            )

            rows = session.execute(stmt).mappings().all()

            if not rows:
                return None

            records = []
            for r in rows:
                record = {
                    "ticker": r["ticker"],
                    "date": pd.Timestamp(r["date"], tz="UTC"),
                }
                fd = r["feature_data"]
                if isinstance(fd, dict):
                    record.update(fd)
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

        if df is None or df.empty:
            return 0

        available_cols = [c for c in feature_columns if c in df.columns]

        if not available_cols:
            logger.warning("store_features: no matching feature columns found")
            return 0

        work = df[["ticker", "date"] + available_cols].copy()

        feat_block = (
            work[available_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        feat_dicts = feat_block.to_dict("records")

        tickers_upper = work["ticker"].str.upper().tolist()
        dates = pd.to_datetime(work["date"]).dt.date.tolist()

        records = [
            {
                "ticker": tickers_upper[i],
                "date": dates[i],
                "feature_version": feature_version,
                "feature_data": feat_dicts[i],
            }
            for i in range(len(work))
        ]

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

        with get_session() as session:

            stmt = delete(ComputedFeature).where(
                ComputedFeature.feature_version == feature_version
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
    Storage and retrieval of model predictions with agent tracking and outcomes.

    FIX (Issue #25): Extended to store individual agent outputs.
    NEW (Issue #26): Added outcome computation methods.
    NEW (Issue #27): Added agent performance methods.
    """

    @staticmethod
    def store_predictions(predictions: List[dict]) -> int:
        """Store predictions with agent tracking fields (Issue #25)."""

        if not predictions:
            return 0

        pred_df = pd.DataFrame(predictions)

        # ─── Core Fields ─────────────────────
        pred_df["ticker"] = pred_df["ticker"].str.upper()
        pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.date
        pred_df["model_version"] = (
            pred_df.get("model_version", "unknown").fillna("unknown")
            if "model_version" in pred_df
            else "unknown"
        )
        pred_df["schema_signature"] = (
            pred_df.get("schema_signature", "unknown").fillna("unknown")
            if "schema_signature" in pred_df
            else "unknown"
        )
        pred_df["raw_model_score"] = pd.to_numeric(
            pred_df.get("raw_model_score", 0), errors="coerce"
        ).fillna(0.0)
        pred_df["hybrid_score"] = pd.to_numeric(
            pred_df.get("hybrid_score", 0), errors="coerce"
        ).fillna(0.0)
        pred_df["weight"] = pd.to_numeric(
            pred_df.get("weight", 0), errors="coerce"
        ).fillna(0.0)
        pred_df["signal"] = (
            pred_df.get("signal", pd.Series(dtype=str)).where(
                pred_df.get("signal", pd.Series(dtype=str)).notna(), None
            )
            if "signal" in pred_df
            else None
        )
        pred_df["drift_state"] = (
            pred_df.get("drift_state", pd.Series(dtype=str)).where(
                pred_df.get("drift_state", pd.Series(dtype=str)).notna(), None
            )
            if "drift_state" in pred_df
            else None
        )

        # ─── Agent Fields ────────────────────
        def _safe_get(row, *keys, default=None):
            try:
                val = row
                for key in keys:
                    if isinstance(val, dict):
                        val = val.get(key)
                    else:
                        return default
                    if val is None:
                        return default
                return val
            except Exception:
                return default

        pred_df["signal_agent_score"] = pred_df.apply(
            lambda r: _safe_get(r, "agents", "signal_agent", "score"), axis=1
        )
        pred_df["signal_agent_signal"] = pred_df.apply(
            lambda r: _safe_get(r, "agents", "signal_agent", "signal"), axis=1
        )
        pred_df["signal_agent_confidence"] = pred_df.apply(
            lambda r: _safe_get(r, "agents", "signal_agent", "confidence_numeric"),
            axis=1,
        )
        pred_df["signal_agent_risk_level"] = pred_df.apply(
            lambda r: _safe_get(r, "agents", "signal_agent", "risk_level"), axis=1
        )

        pred_df["technical_agent_score"] = pred_df.apply(
            lambda r: _safe_get(r, "agents", "technical_agent", "score"), axis=1
        )
        pred_df["technical_agent_bias"] = pred_df.apply(
            lambda r: _safe_get(r, "agents", "technical_agent", "bias"), axis=1
        )
        pred_df["technical_agent_volatility_regime"] = pred_df.apply(
            lambda r: _safe_get(
                r, "agents", "technical_agent", "signals", "volatility_regime"
            )
            or _safe_get(r, "agents", "technical_agent", "volatility_regime"),
            axis=1,
        )

        pred_df["political_agent_score"] = pred_df.apply(
            lambda r: _safe_get(r, "political_risk_score"), axis=1
        )
        pred_df["political_agent_label"] = pred_df.apply(
            lambda r: _safe_get(r, "political_risk_label"), axis=1
        )

        # ─── Build Records ───────────────────
        keep_cols = [
            "ticker", "date", "model_version", "schema_signature",
            "raw_model_score", "hybrid_score", "weight", "signal", "drift_state",
            "signal_agent_score", "signal_agent_signal", "signal_agent_confidence", "signal_agent_risk_level",
            "technical_agent_score", "technical_agent_bias", "technical_agent_volatility_regime",
            "political_agent_score", "political_agent_label",
        ]

        keep_cols = [c for c in keep_cols if c in pred_df.columns]
        records = pred_df[keep_cols].to_dict("records")

        if not records:
            return 0

        with get_session() as session:

            stmt = pg_insert(ModelPrediction).values(records)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_prediction_ticker_date_model"
            )
            result = session.execute(stmt)
            inserted = result.rowcount if result.rowcount else 0

            agent_fields_populated = sum(
                1
                for r in records
                if r.get("signal_agent_score") is not None
                or r.get("technical_agent_score") is not None
            )

            logger.info(
                "DB prediction store | total=%d inserted=%d agent_fields=%d",
                len(records),
                inserted,
                agent_fields_populated,
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

        with get_session() as session:

            stmt = select(ModelPrediction).where(ModelPrediction.date == date)

            if model_version:
                stmt = stmt.where(ModelPrediction.model_version == model_version)

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
                    "signal_agent_score": r.signal_agent_score,
                    "signal_agent_signal": r.signal_agent_signal,
                    "signal_agent_confidence": r.signal_agent_confidence,
                    "signal_agent_risk_level": r.signal_agent_risk_level,
                    "technical_agent_score": r.technical_agent_score,
                    "technical_agent_bias": r.technical_agent_bias,
                    "technical_agent_volatility_regime": r.technical_agent_volatility_regime,
                    "political_agent_score": r.political_agent_score,
                    "political_agent_label": r.political_agent_label,
                    "actual_forward_return": r.actual_forward_return,
                    "direction_correct": r.direction_correct,
                }
                for r in rows
            ]

            return pd.DataFrame(records)

    # ─── Issue #26: Outcome Methods ──────────────────────────

    @staticmethod
    def get_predictions_needing_outcomes(
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Query predictions needing outcome computation (Issue #26)."""
        
        with get_session() as session:
            
            stmt = (
                select(
                    ModelPrediction.id,
                    ModelPrediction.ticker,
                    ModelPrediction.date,
                    ModelPrediction.signal,
                    ModelPrediction.raw_model_score,
                    ModelPrediction.predicted_at,
                )
                .where(
                    ModelPrediction.date >= start_date,
                    ModelPrediction.date <= end_date,
                    ModelPrediction.outcome_fetched_at.is_(None),
                )
                .order_by(ModelPrediction.date, ModelPrediction.ticker)
            )
            
            rows = session.execute(stmt).mappings().all()
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows)
            
            logger.debug(
                "DB outcome query | start=%s end=%s count=%d",
                start_date,
                end_date,
                len(df),
                extra={
                    "component": "db.repository",
                    "function": "get_predictions_needing_outcomes",
                },
            )
            
            return df

    @staticmethod
    def update_prediction_outcomes(outcomes: List[dict]) -> int:
        """Batch update predictions with outcomes (Issue #26)."""
        
        if not outcomes:
            return 0
        
        with get_session() as session:
            
            updated_count = 0
            
            for outcome in outcomes:
                
                stmt = (
                    update(ModelPrediction)
                    .where(ModelPrediction.id == outcome["id"])
                    .values(
                        actual_forward_return=outcome["actual_forward_return"],
                        direction_correct=outcome["direction_correct"],
                        prediction_error=outcome["prediction_error"],
                        outcome_fetched_at=outcome["outcome_fetched_at"],
                    )
                )
                
                result = session.execute(stmt)
                updated_count += result.rowcount
            
            logger.info(
                "DB outcome update | total=%d updated=%d",
                len(outcomes),
                updated_count,
                extra={
                    "component": "db.repository",
                    "function": "update_prediction_outcomes",
                },
            )
            
            return updated_count

    # ─── Issue #27: Agent Performance Methods ────────────────

    @staticmethod
    def get_predictions_with_outcomes(
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Query predictions with outcomes for agent evaluation (Issue #27)."""
        
        with get_session() as session:
            
            stmt = (
                select(ModelPrediction)
                .where(
                    ModelPrediction.date >= start_date,
                    ModelPrediction.date <= end_date,
                    ModelPrediction.outcome_fetched_at.is_not(None),
                )
                .order_by(ModelPrediction.date)
            )
            
            rows = session.execute(stmt).scalars().all()
            
            if not rows:
                return None
            
            records = [
                {
                    "id": r.id,
                    "ticker": r.ticker,
                    "date": str(r.date),
                    "signal": r.signal,
                    "raw_model_score": r.raw_model_score,
                    "signal_agent_score": r.signal_agent_score,
                    "signal_agent_signal": r.signal_agent_signal,
                    "signal_agent_confidence": r.signal_agent_confidence,
                    "signal_agent_risk_level": r.signal_agent_risk_level,
                    "technical_agent_score": r.technical_agent_score,
                    "technical_agent_bias": r.technical_agent_bias,
                    "technical_agent_volatility_regime": r.technical_agent_volatility_regime,
                    "actual_forward_return": r.actual_forward_return,
                    "direction_correct": r.direction_correct,
                    "prediction_error": r.prediction_error,
                }
                for r in rows
            ]
            
            df = pd.DataFrame(records)
            
            logger.debug(
                "DB agent eval query | start=%s end=%s count=%d",
                start_date,
                end_date,
                len(df),
                extra={
                    "component": "db.repository",
                    "function": "get_predictions_with_outcomes",
                },
            )
            
            return df

    @staticmethod
    def store_agent_performance(records: List[dict]) -> int:
        """Store agent performance metrics (Issue #27)."""
        
        if not records:
            return 0
        
        with get_session() as session:
            
            stmt = pg_insert(AgentPerformance).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_agent_perf_name_date",
                set_={
                    "direction_accuracy": stmt.excluded.direction_accuracy,
                    "sharpe_ratio": stmt.excluded.sharpe_ratio,
                    "num_predictions": stmt.excluded.num_predictions,
                    "avg_score": stmt.excluded.avg_score,
                    "confidence_calibration": stmt.excluded.confidence_calibration,
                    "mean_absolute_error": stmt.excluded.mean_absolute_error,
                },
            )
            
            result = session.execute(stmt)
            upserted = result.rowcount if result.rowcount else 0
            
            logger.info(
                "DB agent performance store | total=%d upserted=%d",
                len(records),
                upserted,
                extra={
                    "component": "db.repository",
                    "function": "store_agent_performance",
                },
            )
            
            return upserted

    @staticmethod
    def get_agent_performance(
        agent_name: Optional[str] = None,
        days: int = 30,
    ) -> Optional[pd.DataFrame]:
        """Query agent performance history (Issue #27)."""
        
        with get_session() as session:
            
            cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=days)
            
            stmt = select(AgentPerformance).where(
                AgentPerformance.evaluation_date >= cutoff.strftime("%Y-%m-%d")
            )
            
            if agent_name:
                stmt = stmt.where(AgentPerformance.agent_name == agent_name)
            
            stmt = stmt.order_by(
                AgentPerformance.agent_name,
                AgentPerformance.evaluation_date,
            )
            
            rows = session.execute(stmt).scalars().all()
            
            if not rows:
                return None
            
            records = [
                {
                    "agent_name": r.agent_name,
                    "evaluation_date": str(r.evaluation_date),
                    "direction_accuracy": r.direction_accuracy,
                    "sharpe_ratio": r.sharpe_ratio,
                    "num_predictions": r.num_predictions,
                    "avg_score": r.avg_score,
                    "confidence_calibration": r.confidence_calibration,
                    "mean_absolute_error": r.mean_absolute_error,
                }
                for r in rows
            ]
            
            return pd.DataFrame(records)