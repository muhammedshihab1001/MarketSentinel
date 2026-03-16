# =========================================================
# INSTITUTIONAL INFERENCE PIPELINE v5.0
# Stable | Drift-Aware | CV-Optimized | Noise-Controlled
# =========================================================

import time
import threading
import numpy as np
import pandas as pd
import logging
import os
from typing import List

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
from core.agent.political_risk_agent import PoliticalRiskAgent

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    PIPELINE_FAILURES,
    INFERENCE_IN_PROGRESS,
    SIGNAL_DISTRIBUTION
)

logger = logging.getLogger("marketsentinel.pipeline")

EPSILON = 1e-12

_SHARED_MODEL_LOADER = None
_MODEL_LOCK = threading.Lock()


# =========================================================
# SHARED MODEL LOADER
# =========================================================

def get_shared_model_loader():

    global _SHARED_MODEL_LOADER

    if _SHARED_MODEL_LOADER is None:

        with _MODEL_LOCK:

            if _SHARED_MODEL_LOADER is None:

                logger.info("Initializing shared ModelLoader")

                _SHARED_MODEL_LOADER = ModelLoader()

                # warm load
                _ = _SHARED_MODEL_LOADER.xgb

    return _SHARED_MODEL_LOADER


# =========================================================
# INFERENCE PIPELINE
# =========================================================

class InferencePipeline:

    TARGET_GROSS_EXPOSURE = 1.0
    MIN_UNIVERSE_WIDTH = int(os.getenv("MIN_UNIVERSE_WIDTH", "8"))

    MIN_SCORE_STD = 1e-6
    MAX_POSITION_WEIGHT = 0.20

    TOP_K = 10
    BOTTOM_K = 10
    TOP_SELECTION = 5

    SCORE_WINSOR_Q = 0.02
    BASE_LIQUIDITY = 5e5

    SNAPSHOT_CACHE_TTL = int(os.getenv("SNAPSHOT_CACHE_TTL", "120"))
    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))

    # -----------------------------------------------------

    def __init__(self):

        self.market_data = MarketDataService()

        self.models = get_shared_model_loader()

        self.cache = RedisCache()

        self.drift_detector = DriftDetector()

        self.signal_agent = SignalAgent()
        self.technical_agent = TechnicalRiskAgent()
        self.portfolio_agent = PortfolioDecisionAgent()
        self.political_agent = PoliticalRiskAgent()

        self._validate_models_loaded()

    # =========================================================
    # MODEL VALIDATION
    # =========================================================

    def _validate_models_loaded(self):

        container = self.models._xgb_container

        if container is None:
            raise RuntimeError("Model container not initialized.")

        if container.schema_signature != get_schema_signature():
            raise RuntimeError("Schema signature mismatch.")

        logger.info("Model verified | version=%s", container.version)

    # =========================================================
    # FEATURE FRAME BUILDER
    # =========================================================

    def _build_cross_sectional_frame(self, tickers: List[str]):

        end_date = pd.Timestamp.utcnow()

        start_date = end_date - pd.Timedelta(days=self.INFERENCE_LOOKBACK_DAYS)

        data, failures = self.market_data.get_price_data_batch(
            tickers,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        if failures:

            logger.warning(
                "Market data partial failure | success=%d | failed=%d",
                len(data),
                len(failures),
            )

        if len(data) < self.MIN_UNIVERSE_WIDTH:

            raise RuntimeError(
                f"Too few tickers available ({len(data)})."
            )

        combined_prices = pd.concat(list(data.values()), ignore_index=True)

        combined_prices = combined_prices.dropna(subset=["close"])

        combined = FeatureEngineer.build_feature_pipeline(
            combined_prices,
            training=False
        )

        if combined.empty:
            raise RuntimeError("Feature generation produced empty dataset.")

        logger.info(
            "Cross-sectional frame built | rows=%d tickers=%d",
            len(combined),
            combined["ticker"].nunique(),
        )

        return combined

    # =========================================================
    # STABLE SOFTMAX
    # =========================================================

    def _stable_softmax(self, x):

        x = np.asarray(x)

        x = x - np.max(x)

        e = np.exp(x)

        return e / (np.sum(e) + EPSILON)

    # =========================================================
    # PORTFOLIO CONSTRUCTION
    # =========================================================

    def _construct_portfolio_from_rows(self, longs, shorts):

        weights = {}

        if longs:

            scores = np.array([x["hybrid_consensus_score"] for x in longs])

            soft = self._stable_softmax(scores)

            for row, w in zip(longs, soft):

                weights[row["ticker"]] = min(
                    w * self.TARGET_GROSS_EXPOSURE / 2,
                    self.MAX_POSITION_WEIGHT
                )

        if shorts:

            scores = np.array([x["hybrid_consensus_score"] for x in shorts])

            soft = self._stable_softmax(-scores)

            for row, w in zip(shorts, soft):

                weights[row["ticker"]] = -min(
                    w * self.TARGET_GROSS_EXPOSURE / 2,
                    self.MAX_POSITION_WEIGHT
                )

        return weights

    # =========================================================
    # WINSORIZE
    # =========================================================

    def _winsorize(self, x):

        if len(x) < 3:
            return x

        lower = np.quantile(x, self.SCORE_WINSOR_Q)

        upper = np.quantile(x, 1 - self.SCORE_WINSOR_Q)

        return np.clip(x, lower, upper)

    # =========================================================
    # DRIFT SAFE CALL
    # =========================================================

    def _safe_drift(self, feature_df):

        try:
            return self.drift_detector.detect(feature_df)

        except Exception:

            logger.warning("Drift detector failed.")

            return {
                "drift_state": "unknown",
                "severity_score": 0.0,
                "exposure_scale": 1.0
            }

    # =========================================================
    # SAFE AGENT EXECUTION
    # =========================================================

    def _safe_agent(self, agent, context):

        try:
            return agent.analyze(context)

        except Exception:

            logger.exception("Agent failure.")

            return {"score": 0.0, "agent_score": 0.0, "hybrid": {"score": 0.0}}

    # =========================================================
    # MAIN SNAPSHOT
    # =========================================================

    def run_snapshot(self, tickers: List[str]):

        container = self.models._xgb_container
        model = self.models.xgb

        full_universe = list(MarketUniverse.get_universe())

        requested = sorted(set(tickers)) if tickers else full_universe

        cache_key = self.cache.build_key({
            "type": "snapshot",
            "requested": requested,
            "model_version": container.version,
            "schema": container.schema_signature
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

            feature_df = validate_feature_schema(
                latest_df.loc[:, MODEL_FEATURES],
                mode="inference"
            ).astype(DTYPE)

            drift_result = self._safe_drift(feature_df)

            exposure_scale = drift_result.get("exposure_scale", 1.0)

            feature_np = np.ascontiguousarray(feature_df)

            raw_scores = model.predict(feature_np)

            raw_scores = self._winsorize(raw_scores)

            score_std = raw_scores.std()

            if score_std < self.MIN_SCORE_STD:
                raise RuntimeError("Model scores collapsed.")

            raw_scores = (
                raw_scores - raw_scores.mean()
            ) / (score_std + EPSILON)

            latest_df["raw_model_score"] = raw_scores

            snapshot_rows = []

            for _, row in latest_df.iterrows():

                direction = "LONG" if row["raw_model_score"] > 0 else "SHORT"

                row_dict = {**row.to_dict(), "signal": direction}

                context = {"row": row_dict}

                signal_output = self._safe_agent(self.signal_agent, context)

                technical_output = self._safe_agent(self.technical_agent, context)

                hybrid_score = (
                    signal_output.get("hybrid", {}).get("score", 0) +
                    technical_output.get("score", 0)
                ) / 2

                snapshot_rows.append({
                    "date": str(row["date"]),
                    "ticker": row["ticker"],
                    "raw_model_score": float(row["raw_model_score"]),
                    "hybrid_consensus_score": float(hybrid_score),
                    "weight": 0.0
                })

            ranked = sorted(
                snapshot_rows,
                key=lambda x: x["hybrid_consensus_score"]
            )

            longs = ranked[-min(self.TOP_K, len(ranked)):]
            shorts = ranked[:min(self.BOTTOM_K, len(ranked))]

            weights = self._construct_portfolio_from_rows(longs, shorts)

            for row in snapshot_rows:

                base_weight = weights.get(row["ticker"], 0.0)

                row["weight"] = float(base_weight * exposure_scale)

            result = {
                "snapshot_date": str(latest_date),
                "model_version": container.version,
                "drift": drift_result,
                "signals": snapshot_rows
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