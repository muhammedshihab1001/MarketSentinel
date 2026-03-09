# =========================================================
# INSTITUTIONAL INFERENCE PIPELINE v4.6
# Stable | Drift-Aware | CV-Optimized | Noise-Controlled
# Backward-Compatible Portfolio API
# =========================================================

import time
import threading
import numpy as np
import pandas as pd
import logging
import os
from typing import List, Dict, Any

from core.data.market_data_service import MarketDataService
from core.features.feature_engineering import FeatureEngineer
from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature,
    DTYPE,
)
from core.market.universe import MarketUniverse

from app.inference.model_loader import ModelLoader
from app.inference.cache import RedisCache
from core.agent.signal_agent import SignalAgent
from core.agent.technical_risk_agent import TechnicalRiskAgent
from core.agent.portfolio_decision_agent import PortfolioDecisionAgent
from core.agent.political_risk_agent import PoliticalRiskAgent   # NEW

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    PIPELINE_FAILURES,
    INFERENCE_IN_PROGRESS
)

logger = logging.getLogger("marketsentinel.pipeline")

EPSILON = 1e-12
_SHARED_MODEL_LOADER = None
_MODEL_LOCK = threading.Lock()


def get_shared_model_loader():
    global _SHARED_MODEL_LOADER
    if _SHARED_MODEL_LOADER is None:
        with _MODEL_LOCK:
            if _SHARED_MODEL_LOADER is None:
                logger.info("Initializing shared ModelLoader")
                _SHARED_MODEL_LOADER = ModelLoader()
                _ = _SHARED_MODEL_LOADER.xgb
    return _SHARED_MODEL_LOADER


class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    MIN_UNIVERSE_WIDTH = 8
    MIN_SCORE_STD = 1e-6
    MAX_POSITION_WEIGHT = 0.20

    TOP_K = 10
    BOTTOM_K = 10
    TOP_SELECTION = 5

    SCORE_WINSOR_Q = 0.02
    BASE_LIQUIDITY = 5e5

    SNAPSHOT_CACHE_TTL = 5
    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))

    # ---------------------------------------------------------

    def __init__(self):
        self.market_data = MarketDataService()
        self.models = get_shared_model_loader()
        self.cache = RedisCache()
        self.drift_detector = DriftDetector()

        self.signal_agent = SignalAgent()
        self.technical_agent = TechnicalRiskAgent()
        self.portfolio_agent = PortfolioDecisionAgent()

        # NEW AGENT
        self.political_agent = PoliticalRiskAgent()

        self._validate_models_loaded()

    # ---------------------------------------------------------

    def _validate_models_loaded(self):
        if self.models.schema_signature != get_schema_signature():
            raise RuntimeError("Schema signature mismatch.")
        logger.info("Model verified | version=%s", self.models.xgb_version)

    # ---------------------------------------------------------

    def _winsorize(self, x):
        if len(x) < 3:
            return x
        lower = np.quantile(x, self.SCORE_WINSOR_Q)
        upper = np.quantile(x, 1 - self.SCORE_WINSOR_Q)
        return np.clip(x, lower, upper)

    def _softmax(self, x):
        if len(x) == 0:
            return np.array([])
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + EPSILON)

    # ---------------------------------------------------------
    # BACKWARD COMPATIBILITY FIX (FOR EQUITY ROUTE)
    # ---------------------------------------------------------

    def _construct_portfolio(self, longs, shorts):
        return self._construct_portfolio_from_rows(longs, shorts)

    # ---------------------------------------------------------

    def _build_agent_context(
        self,
        row: Dict[str, Any],
        probability_stats: Dict[str, Any],
        drift_result: Dict[str, Any],
        political_risk_label: str   # NEW
    ) -> Dict[str, Any]:
        return {
            "row": row,
            "probability_stats": probability_stats,
            "drift_score": drift_result.get("severity_score", 0),
            "drift_state": drift_result.get("drift_state"),
            "political_risk_label": political_risk_label  # NEW
        }

    # =========================================================
    # MAIN SNAPSHOT
    # =========================================================

    def run_snapshot(self, tickers: List[str]):

        full_universe = list(MarketUniverse.get_universe())
        if not full_universe:
            raise RuntimeError("Universe empty.")

        requested = sorted(set(tickers)) if tickers else full_universe

        cache_key = self.cache.build_key({
            "type": "snapshot",
            "requested": requested
        })

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        INFERENCE_IN_PROGRESS.inc()
        start_time = time.time()

        try:

            df = self._build_cross_sectional_frame(full_universe)

            latest_date = df["date"].max()
            latest_df = df[df["date"] == latest_date].copy()
            latest_df = latest_df[latest_df["ticker"].isin(requested)]

            latest_df = latest_df.replace([np.inf, -np.inf], np.nan)
            latest_df = latest_df.dropna(subset=MODEL_FEATURES)

            if latest_df.empty:
                raise RuntimeError("Latest snapshot invalid.")

            if "dollar_volume" in latest_df.columns:
                q25 = latest_df["dollar_volume"].quantile(0.25)
                liquidity_threshold = max(self.BASE_LIQUIDITY, q25)
                filtered = latest_df[
                    latest_df["dollar_volume"] >= liquidity_threshold
                ]
                if len(filtered) >= self.MIN_UNIVERSE_WIDTH:
                    latest_df = filtered

            feature_df = validate_feature_schema(
                latest_df.loc[:, MODEL_FEATURES],
                mode="inference"
            ).astype(DTYPE)

            drift_result = self._safe_drift(feature_df)
            exposure_scale = drift_result.get("exposure_scale", 1.0)

            # -----------------------------------------------------
            # POLITICAL RISK FETCH (ONCE PER SNAPSHOT)
            # -----------------------------------------------------

            political_result = self.political_agent.get_political_risk("GLOBAL")
            political_label = political_result.get("risk_label", "LOW")

            raw_scores = self.models.xgb.predict(feature_df)
            score_std = float(np.std(raw_scores))

            if score_std < self.MIN_SCORE_STD:
                logger.warning("Low dispersion detected in snapshot scores.")
                return {
                    "snapshot_date": str(latest_date),
                    "signals": [],
                    "warning": "Low score dispersion — snapshot skipped."
                }

            raw_scores = self._winsorize(raw_scores)
            raw_scores = (
                raw_scores - raw_scores.mean()
            ) / (raw_scores.std() + EPSILON)

            latest_df["raw_model_score"] = raw_scores

            probability_stats = {
                "mean": float(np.mean(raw_scores)),
                "std": float(np.std(raw_scores))
            }

            snapshot_rows = []

            for _, row in latest_df.iterrows():

                direction = "LONG" if row["raw_model_score"] > 0 else "SHORT"
                row_dict = {**row.to_dict(), "signal": direction}

                context = self._build_agent_context(
                    row=row_dict,
                    probability_stats=probability_stats,
                    drift_result=drift_result,
                    political_risk_label=political_label
                )

                signal_output = self.signal_agent.analyze(context)
                technical_output = self.technical_agent.analyze(context)

                total_weight = (
                    self.signal_agent.weight +
                    self.technical_agent.weight
                )

                hybrid_score = (
                    signal_output["hybrid"]["score"] * self.signal_agent.weight +
                    technical_output["score"] * self.technical_agent.weight
                ) / total_weight

                snapshot_rows.append({
                    "date": str(row["date"]),
                    "ticker": row["ticker"],
                    "raw_model_score": float(row["raw_model_score"]),
                    "agent_score": float(signal_output["agent_score"]),
                    "technical_score": float(technical_output["score"]),
                    "hybrid_consensus_score": float(hybrid_score),
                    "weight": 0.0,
                    "agents": {
                        "signal_agent": signal_output,
                        "technical_agent": technical_output
                    }
                })

            ranked = sorted(snapshot_rows, key=lambda x: x["hybrid_consensus_score"])

            longs = ranked[-min(self.TOP_K, len(ranked)):]
            shorts = ranked[:min(self.BOTTOM_K, len(ranked))]

            weights = self._construct_portfolio_from_rows(longs, shorts)

            for row in snapshot_rows:
                base_weight = weights.get(row["ticker"], 0.0)
                row["weight"] = float(base_weight * exposure_scale)

            top_5 = sorted(
                snapshot_rows,
                key=lambda x: x["hybrid_consensus_score"],
                reverse=True
            )[:min(self.TOP_SELECTION, len(snapshot_rows))]

            snapshot_core = {
                "snapshot_date": str(latest_date),
                "model_version": self.models.xgb_version,
                "schema_version": self.models.schema_version,
                "gross_exposure": float(sum(abs(x["weight"]) for x in snapshot_rows)),
                "net_exposure": float(sum(x["weight"] for x in snapshot_rows)),
                "drift": drift_result,
                "signals": snapshot_rows
            }

            decision_report = self.portfolio_agent.analyze_snapshot(snapshot_core)

            governance = {
                "baseline_status": self.models.baseline_status,
                "schema_signature": self.models.schema_signature,
                "dataset_hash": self.models.dataset_hash
            }

            result = {
                **snapshot_core,
                "universe_size": int(len(latest_df)),
                "score_mean": probability_stats["mean"],
                "score_std": probability_stats["std"],
                "top_5": top_5,
                "decision_report": decision_report,
                "governance": governance,
                "snapshot_quality": "stable"
            }

            self.cache.set(cache_key, result, ex=self.SNAPSHOT_CACHE_TTL)

            MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
            MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(
                time.time() - start_time
            )

            return result

        except Exception:
            PIPELINE_FAILURES.labels(stage="snapshot").inc()
            logger.exception("Snapshot failure.")
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()