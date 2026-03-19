# =========================================================
# INSTITUTIONAL INFERENCE PIPELINE v5.5
#
# Changes from v5.4:
#   FIX 1: snapshot_rows now include "agents" dict so
#           /agent/explain and PortfolioDecisionAgent work.
#   FIX 2: TOP_K / BOTTOM_K reduced from 10 → 5 for better
#           selectivity with 30-100 ticker universe.
#   FIX 3: run_snapshot checks fixed background cache key
#           ("ms:background_snapshot:latest") before running
#           full inference — matches main.py background loop.
#   FIX 4: hybrid_score now includes political_agent output
#           weighted correctly across all three agents.
# =========================================================

import time
import threading
import numpy as np
import pandas as pd
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

from core.db.repository import PredictionRepository
from core.logging.logger import get_logger

from app.monitoring.metrics import (
    MODEL_INFERENCE_COUNT,
    MODEL_INFERENCE_LATENCY,
    PIPELINE_FAILURES,
    INFERENCE_IN_PROGRESS,
)

logger = get_logger("marketsentinel.pipeline")

EPSILON = 1e-12

# Fixed Redis key written by main.py background loop.
# live-snapshot checks this first so it never re-runs inference
# when the background loop already has a fresh result.
BACKGROUND_SNAPSHOT_KEY = "ms:background_snapshot:latest"

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

    # Reduced from 10 → 5: with 100 tickers, top-5 long + bottom-5 short
    # = 10% of universe selected. Much more selective signal.
    TOP_K = int(os.getenv("PIPELINE_TOP_K", "5"))
    BOTTOM_K = int(os.getenv("PIPELINE_BOTTOM_K", "5"))

    SCORE_WINSOR_Q = 0.02

    SNAPSHOT_CACHE_TTL = int(os.getenv("SNAPSHOT_CACHE_TTL", "120"))
    INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))

    STORE_PREDICTIONS = os.getenv("STORE_PREDICTIONS", "1") == "1"

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
    # DATE HELPERS
    # =========================================================

    def _get_date_window(self):
        """
        Compute start_date / end_date from INFERENCE_LOOKBACK_DAYS.
        MarketDataService.get_price_data_batch() requires these.
        """
        end_date = pd.Timestamp.now(tz="UTC")
        start_date = end_date - pd.Timedelta(days=self.INFERENCE_LOOKBACK_DAYS)
        return (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    # =========================================================
    # FEATURE FRAME BUILDER
    # =========================================================

    def _build_cross_sectional_frame(self, tickers: List[str]):
        """
        Build full cross-sectional feature frame from PostgreSQL.
        Passes start_date/end_date computed from INFERENCE_LOOKBACK_DAYS.
        """

        start_date, end_date = self._get_date_window()

        data, failures = self.market_data.get_price_data_batch(
            tickers,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            min_history=min(self.INFERENCE_LOOKBACK_DAYS, 60),
        )

        if failures:
            logger.warning(
                "Market data partial failure | success=%d | failed=%d",
                len(data),
                len(failures),
            )

        if len(data) < self.MIN_UNIVERSE_WIDTH:
            raise RuntimeError(f"Too few tickers available ({len(data)}).")

        combined_prices = pd.concat(list(data.values()), ignore_index=True)
        combined_prices = combined_prices.dropna(subset=["close"])

        combined = FeatureEngineer.build_feature_pipeline(
            combined_prices,
            training=False,
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
    # WINSORIZE
    # =========================================================

    def _winsorize(self, x):

        if len(x) < 3:
            return x

        lower = np.quantile(x, self.SCORE_WINSOR_Q)
        upper = np.quantile(x, 1 - self.SCORE_WINSOR_Q)

        return np.clip(x, lower, upper)

    # =========================================================
    # SAFE DRIFT
    # =========================================================

    def _safe_drift(self, feature_df):

        try:
            return self.drift_detector.detect(feature_df)
        except Exception:
            logger.warning("Drift detector failed — using safe defaults.")
            return {
                "drift_state": "unknown",
                "severity_score": 0.0,
                "exposure_scale": 1.0,
            }

    # =========================================================
    # SAFE AGENT
    # =========================================================

    def _safe_agent(self, agent, context):

        try:
            return agent.analyze(context)
        except Exception:
            logger.exception("Agent failure.")
            return {"score": 0.0, "agent_score": 0.0, "hybrid": {"score": 0.0}}

    # =========================================================
    # HYBRID SCORE
    # Combines signal_agent + technical_agent + political_agent.
    # All three return scores on 0-1 scale via agent_score field.
    # Weights: signal=0.5, technical=0.3, political=0.2
    # =========================================================

    def _compute_hybrid_score(self, signal_output, technical_output, political_output):
        """
        Weighted combination of three agent scores.
        All agent_score fields are 0-1 normalized.
        """

        sa = float(signal_output.get("agent_score", 0.0))
        ta = float(technical_output.get("score", 0.0))
        pa = float(political_output.get("agent_score", 0.0))

        # Weight: signal matters most, technical confirms, political gates
        hybrid = 0.5 * sa + 0.3 * ta + 0.2 * pa

        return float(np.clip(hybrid, 0.0, 1.0))

    # =========================================================
    # PORTFOLIO CONSTRUCTION
    # =========================================================

    def _construct_portfolio_from_rows(self, longs, shorts):

        weights = {}
        half_exposure = self.TARGET_GROSS_EXPOSURE / 2.0

        if longs:
            long_weight = min(half_exposure / len(longs), self.MAX_POSITION_WEIGHT)
            for row in longs:
                weights[row["ticker"]] = round(long_weight, 6)

        if shorts:
            short_weight = min(half_exposure / len(shorts), self.MAX_POSITION_WEIGHT)
            for row in shorts:
                weights[row["ticker"]] = round(-short_weight, 6)

        return weights

    # =========================================================
    # STORE PREDICTIONS
    # =========================================================

    def _store_predictions(self, snapshot_rows, drift_result, container):

        if not self.STORE_PREDICTIONS:
            return

        try:

            from datetime import date

            predictions = []

            for row in snapshot_rows:

                raw_date = row.get("date", "")
                try:
                    if isinstance(raw_date, str):
                        parsed_date = pd.Timestamp(raw_date).date()
                    elif hasattr(raw_date, "date"):
                        parsed_date = raw_date.date()
                    else:
                        parsed_date = date.today()
                except Exception:
                    parsed_date = date.today()

                weight = float(row.get("weight", 0.0))
                signal = "LONG" if weight > 0 else ("SHORT" if weight < 0 else "NEUTRAL")

                predictions.append({
                    "ticker": str(row["ticker"]),
                    "date": parsed_date,
                    "model_version": str(container.version),
                    "schema_signature": str(container.schema_signature),
                    "raw_score": float(row.get("raw_model_score", 0.0)),
                    "hybrid_score": float(row.get("hybrid_consensus_score", 0.0)),
                    "weight": weight,
                    "signal": signal,
                    "drift_state": str(drift_result.get("drift_state", "unknown")),
                })

            if predictions:
                PredictionRepository.store_predictions(predictions)
                logger.info(
                    "Predictions stored | count=%d | model=%s",
                    len(predictions),
                    container.version,
                )

        except Exception as e:
            logger.warning(
                "Failed to store predictions (non-blocking) | error=%s", str(e)
            )

    # =========================================================
    # MAIN SNAPSHOT
    # =========================================================

    def run_snapshot(self, tickers: List[str]):

        container = self.models._xgb_container
        model = self.models.xgb

        full_universe = list(MarketUniverse.get_universe())
        requested = sorted(set(tickers)) if tickers else full_universe

        # ----------------------------------------------------------
        # FIX: Check fixed background snapshot key FIRST.
        # main.py _background_snapshot_loop writes to this key.
        # If it's there, serve it instantly — no inference needed.
        # Dynamic build_key() below is a secondary cache for when
        # the background key doesn't exist yet (first boot).
        # ----------------------------------------------------------
        bg_cached = self.cache.get(BACKGROUND_SNAPSHOT_KEY)
        if bg_cached and isinstance(bg_cached, dict) and "signals" in bg_cached:
            # Filter signals to requested tickers if needed
            if set(requested) != set(full_universe):
                bg_cached = dict(bg_cached)
                bg_cached["signals"] = [
                    s for s in bg_cached["signals"]
                    if s.get("ticker") in requested
                ]
            logger.info("Snapshot served from background cache key")
            return bg_cached

        # Secondary cache key (model/schema/ticker fingerprinted)
        cache_key = self.cache.build_key({
            "type": "snapshot",
            "requested": requested,
            "model_version": container.version,
            "schema": container.schema_signature,
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
            latest_df = latest_df.dropna(subset=MODEL_FEATURES, how="any")

            if latest_df.empty:
                raise RuntimeError("Latest snapshot invalid.")

            feature_df = validate_feature_schema(
                latest_df.loc[:, MODEL_FEATURES],
                mode="inference",
            ).astype(DTYPE)

            drift_result = self._safe_drift(feature_df)
            exposure_scale = drift_result.get("exposure_scale", 1.0)

            raw_scores = model.predict(feature_df)
            raw_scores = self._winsorize(raw_scores)

            score_std = raw_scores.std()
            if score_std < self.MIN_SCORE_STD:
                logger.warning("Score collapse detected.")
                score_std = 1.0

            raw_scores = (raw_scores - raw_scores.mean()) / (score_std + EPSILON)

            latest_df = latest_df.copy()
            latest_df["raw_model_score"] = raw_scores

            snapshot_rows = []

            for _, row in latest_df.iterrows():

                direction = "LONG" if row["raw_model_score"] > 0 else "SHORT"
                row_dict = {**row.to_dict(), "signal": direction}
                context = {
                    "row": row_dict,
                    "drift_state": drift_result.get("drift_state"),
                }

                signal_output = self._safe_agent(self.signal_agent, context)
                technical_output = self._safe_agent(self.technical_agent, context)
                political_output = self._safe_agent(self.political_agent, context)

                hybrid_score = self._compute_hybrid_score(
                    signal_output, technical_output, political_output
                )

                # --------------------------------------------------
                # FIX: Write agents into each row.
                # agent.py _explain_logic reads row.get("agents")
                # PortfolioDecisionAgent reads stock.get("agents")
                # Both were always getting empty dicts before this fix.
                # --------------------------------------------------
                snapshot_rows.append({
                    "date": str(row["date"]),
                    "ticker": row["ticker"],
                    "raw_model_score": float(row["raw_model_score"]),
                    "hybrid_consensus_score": float(hybrid_score),
                    "weight": 0.0,
                    "agents": {
                        "signal_agent": signal_output,
                        "technical_agent": technical_output,
                        "political_agent": political_output,
                    },
                })

            ranked = sorted(
                snapshot_rows, key=lambda x: x["hybrid_consensus_score"]
            )
            longs = ranked[-min(self.TOP_K, len(ranked)):]
            shorts = ranked[: min(self.BOTTOM_K, len(ranked))]

            weights = self._construct_portfolio_from_rows(longs, shorts)

            for row in snapshot_rows:
                base_weight = weights.get(row["ticker"], 0.0)
                row["weight"] = float(base_weight * exposure_scale)

            result = {
                "snapshot_date": str(latest_date),
                "model_version": container.version,
                "drift": drift_result,
                "signals": snapshot_rows,
            }

            self._store_predictions(snapshot_rows, drift_result, container)
            self.cache.set(cache_key, result, ex=self.SNAPSHOT_CACHE_TTL)

            MODEL_INFERENCE_COUNT.labels(model="xgboost").inc()
            MODEL_INFERENCE_LATENCY.labels(model="xgboost").observe(
                time.time() - start_time
            )

            logger.info(
                "Snapshot complete | tickers=%d model=%s latency=%.2fs",
                len(snapshot_rows),
                container.version,
                round(time.time() - start_time, 2),
            )

            return result

        except Exception:
            PIPELINE_FAILURES.labels(stage="snapshot").inc()
            logger.exception("Snapshot failure.")
            raise

        finally:
            INFERENCE_IN_PROGRESS.dec()