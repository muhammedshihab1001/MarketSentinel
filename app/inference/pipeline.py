# =========================================================
# INSTITUTIONAL INFERENCE PIPELINE v5.7
#
# FIX (issue 1): __init__ args now optional with defaults.
#   predict.py calls InferencePipeline() with no args.
#   Model + cache are now injected via init_pipeline() in
#   predict.py and stored as module-level singletons.
# FIX: MarketDataService() called without session_factory —
#   it uses the global SQLAlchemy engine, no factory needed.
# FIX: run_snapshot() signature — tickers loaded internally
#   from MarketUniverse, not passed as argument.
# =========================================================

import time
import logging
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from core.schema.feature_schema import MODEL_FEATURES, validate_feature_schema, DTYPE
from core.market.universe import MarketUniverse

logger = logging.getLogger("marketsentinel.pipeline")

PIPELINE_TIMEOUT = float(os.getenv("PIPELINE_TIMEOUT_SECONDS", "12"))
TOP_K = int(os.getenv("PIPELINE_TOP_K", "5"))
BOTTOM_K = int(os.getenv("PIPELINE_BOTTOM_K", "5"))
INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))


class InferencePipeline:

    def __init__(self, model=None, cache=None, db_session_factory=None):
        """
        FIX: All args optional — predict.py calls InferencePipeline()
        with no args. Model is accessed via module-level _model_loader
        set by init_pipeline() in predict.py.
        """
        self._model = model          # ModelLoader instance or None
        self._cache = cache          # RedisCache instance or None
        self._db_session_factory = db_session_factory  # unused — MarketDataService uses global engine

        # Agents — lazy init
        self._signal_agent = None
        self._technical_agent = None
        self._portfolio_agent = None
        self._political_agent = None

    # =====================================================
    # MODEL ACCESSOR
    # Falls back to module-level _model_loader set by init_pipeline()
    # =====================================================

    def _get_model(self):
        """Get the model loader, preferring injected instance."""
        if self._model is not None:
            return self._model
        # Fallback: read from predict.py module-level singleton
        try:
            from app.api.routes.predict import _model_loader
            if _model_loader is not None:
                return _model_loader
        except ImportError:
            pass
        # Last resort: get from model_loader module
        from app.inference.model_loader import get_model_loader
        return get_model_loader()

    # =====================================================
    # AGENT ACCESSORS (lazy init)
    # =====================================================

    @property
    def signal_agent(self):
        if self._signal_agent is None:
            from core.agent.signal_agent import SignalAgent
            self._signal_agent = SignalAgent()
        return self._signal_agent

    @property
    def technical_agent(self):
        if self._technical_agent is None:
            from core.agent.technical_risk_agent import TechnicalRiskAgent
            self._technical_agent = TechnicalRiskAgent()
        return self._technical_agent

    @property
    def portfolio_agent(self):
        if self._portfolio_agent is None:
            from core.agent.portfolio_decision_agent import PortfolioDecisionAgent
            self._portfolio_agent = PortfolioDecisionAgent()
        return self._portfolio_agent

    @property
    def political_agent(self):
        if self._political_agent is None:
            from core.agent.political_risk_agent import PoliticalRiskAgent
            self._political_agent = PoliticalRiskAgent()
        return self._political_agent

    # =====================================================
    # SAFE AGENT CALL
    # =====================================================

    def _safe_agent(self, agent, context: dict) -> dict:
        try:
            return agent.analyze(context) or {}
        except Exception as e:
            logger.debug("Agent %s failed: %s", type(agent).__name__, e)
            return {}

    # =====================================================
    # BUILD CROSS-SECTIONAL FRAME
    # FIX: MarketDataService() — no session_factory arg needed
    # =====================================================

    def _build_cross_sectional_frame(
        self,
        tickers: List[str],
        end_date: str,
    ) -> Optional[pd.DataFrame]:

        from core.data.market_data_service import MarketDataService
        from core.features.feature_engineering import FeatureEngineer

        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - pd.Timedelta(days=INFERENCE_LOOKBACK_DAYS)
        start_date = start_dt.strftime("%Y-%m-%d")

        # FIX: No session_factory — uses global engine
        svc = MarketDataService()
        engineer = FeatureEngineer()

        try:
            price_result = svc.get_price_data_batch(
                tickers,
                start_date=start_date,
                end_date=end_date,
            )
            # get_price_data_batch returns (price_map, errors) tuple
            if isinstance(price_result, tuple):
                price_data, errors = price_result
                if errors:
                    logger.warning(
                        "Price fetch errors for %d tickers: %s",
                        len(errors), list(errors)[:5],
                    )
            else:
                price_data = price_result

        except Exception as e:
            logger.error("Price data fetch failed: %s", e)
            return None

        if not price_data:
            logger.warning("No price data returned for any ticker")
            return None

        # Build a combined multi-ticker frame for cross-sectional features
        all_frames = []
        for ticker, df in price_data.items():
            if df is None or df.empty:
                continue
            # Ensure ticker column
            df = df.copy()
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            all_frames.append(df)

        if not all_frames:
            return None

        combined_prices = pd.concat(all_frames, ignore_index=True)

        # Run feature pipeline on the full cross-section (training=False → uses cache)
        try:
            features = engineer.build_feature_pipeline(combined_prices, training=False)
        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            return None

        return features

    # =====================================================
    # RUN SNAPSHOT
    # FIX: signature takes snapshot_date only — tickers loaded
    # internally from MarketUniverse, NOT passed as arg.
    # Old call: pipeline.run_snapshot(tickers) — WRONG
    # New call: pipeline.run_snapshot() — CORRECT
    # =====================================================

    def run_snapshot(self, snapshot_date: Optional[str] = None) -> dict:
        start_time = time.time()

        if snapshot_date is None:
            snapshot_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        logger.info("Running snapshot | date=%s", snapshot_date)

        # ── Get model ─────────────────────────────────
        try:
            loader = self._get_model()
        except Exception as e:
            return self._error_snapshot(f"Model loader unavailable: {e}")

        # ── Get universe ──────────────────────────────
        try:
            universe = MarketUniverse.snapshot()
            tickers = list(universe.get("tickers", []))
        except Exception as e:
            logger.error("Universe load failed: %s", e)
            return self._error_snapshot("Universe load failed")

        if len(tickers) < int(os.getenv("MIN_UNIVERSE_WIDTH", "8")):
            return self._error_snapshot("Universe too small")

        # ── Build feature frame ───────────────────────
        dataset = self._build_cross_sectional_frame(tickers, snapshot_date)

        if dataset is None or dataset.empty:
            return self._error_snapshot("Feature engineering failed for all tickers")

        # ── Validate features ─────────────────────────
        try:
            available_features = [f for f in MODEL_FEATURES if f in dataset.columns]
            if len(available_features) < len(MODEL_FEATURES) * 0.8:
                logger.warning(
                    "Only %d/%d features available",
                    len(available_features), len(MODEL_FEATURES),
                )
            feature_block = validate_feature_schema(
                dataset.reindex(columns=MODEL_FEATURES, fill_value=0.0),
                mode="inference",
            ).astype(DTYPE)
        except Exception as e:
            logger.error("Feature validation failed: %s", e)
            return self._error_snapshot("Feature validation failed")

        # ── Run XGBoost inference ─────────────────────
        try:
            raw_scores = loader.predict(feature_block)
        except Exception as e:
            logger.error("Model inference failed: %s", e)
            return self._error_snapshot("Model inference failed")

        dataset = dataset.reset_index(drop=True)
        dataset["raw_model_score"] = raw_scores

        # ── Drift detection ───────────────────────────
        drift_state = "none"
        drift_result = {}
        try:
            from core.monitoring.drift_detector import DriftDetector
            detector = DriftDetector()
            drift_result = detector.detect(dataset)
            drift_state = drift_result.get("drift_state", "none")
        except Exception as e:
            logger.warning("Drift detection failed: %s", e)

        # ── Political risk — ONE call for US market ───
        political_output = {}
        try:
            political_output = self._safe_agent(
                self.political_agent,
                {"ticker": "MARKET", "country": "US"},
            )
        except Exception as e:
            logger.debug("Political agent failed: %s", e)

        political_label = political_output.get("political_risk_label", "LOW")

        # ── Per-ticker agent scoring ──────────────────
        snapshot_rows = []
        exposure_scale = float(drift_result.get("exposure_scale", 1.0))

        for idx, row in dataset.iterrows():
            ticker = row.get("ticker", "UNKNOWN")

            # Skip if ticker not in our universe (cross-sectional may include extras)
            if ticker not in tickers:
                continue

            context = {
                "row": row.to_dict(),
                "ticker": ticker,
                "drift_state": drift_state,
                "political_risk_label": political_label,
                "probability_stats": {
                    "mean": float(dataset["raw_model_score"].mean()),
                    "std": float(dataset["raw_model_score"].std()),
                },
            }

            signal_output = self._safe_agent(self.signal_agent, context)
            technical_output = self._safe_agent(self.technical_agent, context)

            raw_score = float(row.get("raw_model_score", 0.0))
            signal_score = float(signal_output.get("score", 0.0))
            technical_score = float(technical_output.get("score", 0.0))

            hybrid_score = float(np.clip(
                0.50 * raw_score + 0.30 * signal_score + 0.20 * technical_score,
                -1.0, 1.0,
            ))

            if political_label == "CRITICAL":
                hybrid_score = 0.0
            elif political_label == "HIGH":
                hybrid_score *= 0.5

            weight = float(np.clip(hybrid_score * exposure_scale, -1.0, 1.0))

            snapshot_rows.append({
                "ticker": ticker,
                "date": str(row.get("date", snapshot_date))[:10],
                "raw_model_score": round(raw_score, 6),
                "hybrid_consensus_score": round(hybrid_score, 6),
                "weight": round(weight, 6),
                "agents": {
                    "signal_agent": signal_output,
                    "technical_agent": technical_output,
                },
            })

        if not snapshot_rows:
            return self._error_snapshot("No valid signals produced")

        snapshot_rows.sort(key=lambda x: x["raw_model_score"], reverse=True)

        long_signals = [r for r in snapshot_rows if r["weight"] > 0.01]
        short_signals = [r for r in snapshot_rows if r["weight"] < -0.01]

        weights = [r["weight"] for r in snapshot_rows]
        gross_exposure = float(sum(abs(w) for w in weights))
        net_exposure = float(sum(weights))

        if gross_exposure > 1.0:
            net_exposure = net_exposure / gross_exposure
            gross_exposure = 1.0

        lc, sc = len(long_signals), len(short_signals)

        if lc > sc * 1.5:
            portfolio_bias = "LONG_BIASED"
        elif sc > lc * 1.5:
            portfolio_bias = "SHORT_BIASED"
        else:
            portfolio_bias = "BALANCED"

        top_5 = snapshot_rows[:TOP_K]
        avg_hybrid = float(np.mean([r["hybrid_consensus_score"] for r in snapshot_rows]))

        portfolio_output = {}
        try:
            portfolio_context = {
                "signals": snapshot_rows,
                "drift_state": drift_state,
                "gross_exposure": round(gross_exposure, 4),
                "net_exposure": round(net_exposure, 4),
            }
            portfolio_output = self._safe_agent(self.portfolio_agent, portfolio_context)
        except Exception as e:
            logger.debug("Portfolio agent failed: %s", e)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        result = {
            "meta": {
                "model_version": getattr(loader, "version", "unknown"),
                "drift_state": drift_state,
                "long_signals": lc,
                "short_signals": sc,
                "avg_hybrid_score": round(avg_hybrid, 6),
                "latency_ms": latency_ms,
            },
            "executive_summary": {
                "top_5_tickers": [r["ticker"] for r in top_5],
                "portfolio_bias": portfolio_bias,
                "gross_exposure": round(gross_exposure, 4),
                "net_exposure": round(net_exposure, 4),
            },
            "snapshot": {
                "snapshot_date": snapshot_date,
                "model_version": getattr(loader, "version", "unknown"),
                "drift": {
                    "drift_detected": drift_result.get("drift_detected", False),
                    "severity_score": drift_result.get("severity_score", 0),
                    "drift_state": drift_state,
                    "exposure_scale": exposure_scale,
                    "drift_confidence": drift_result.get("drift_confidence", 0.0),
                },
                "signals": [
                    {
                        "ticker": r["ticker"],
                        "date": r["date"],
                        "raw_model_score": r["raw_model_score"],
                        "hybrid_consensus_score": r["hybrid_consensus_score"],
                        "weight": r["weight"],
                    }
                    for r in snapshot_rows
                ],
            },
            "_signal_details": {r["ticker"]: r["agents"] for r in snapshot_rows},
            "_political": political_output,
            "_portfolio": portfolio_output,
        }

        logger.info(
            "Snapshot complete | tickers=%d | long=%d | short=%d | latency=%.0fms",
            len(snapshot_rows), lc, sc, latency_ms,
        )

        return result

    # =====================================================
    # ERROR SNAPSHOT
    # =====================================================

    def _error_snapshot(self, reason: str) -> dict:
        logger.error("Snapshot failed: %s", reason)
        return {
            "meta": {
                "model_version": "unknown",
                "drift_state": "none",
                "long_signals": 0,
                "short_signals": 0,
                "avg_hybrid_score": 0.0,
                "latency_ms": 0,
                "error": reason,
            },
            "executive_summary": {
                "top_5_tickers": [],
                "portfolio_bias": "UNKNOWN",
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
            },
            "snapshot": {
                "snapshot_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "model_version": "unknown",
                "drift": {
                    "drift_detected": False,
                    "severity_score": 0,
                    "drift_state": "none",
                    "exposure_scale": 1.0,
                    "drift_confidence": 0.0,
                },
                "signals": [],
            },
            "_signal_details": {},
            "_political": {},
            "_portfolio": {},
        }