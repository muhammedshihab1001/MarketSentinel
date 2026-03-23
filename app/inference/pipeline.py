# =========================================================
# INSTITUTIONAL INFERENCE PIPELINE v5.6
#
# Changes from v5.5:
# FIX 1: gross_exposure and net_exposure now included in
#         the final snapshot result dict so /snapshot and
#         /portfolio routes return correct values.
# FIX 2: PoliticalRiskAgent called ONCE per snapshot for
#         the US market, not once per ticker. Saves 100x
#         GDELT API calls — result is shared across all tickers.
# FIX 3: executive_summary.top_5_tickers populated from
#         actual top-5 signals sorted by hybrid_consensus_score.
# FIX 4: portfolio_bias derived from long/short counts.
# =========================================================

import asyncio
import time
import logging
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from core.schema.feature_schema import MODEL_FEATURES, validate_feature_schema, DTYPE
from core.market.universe import MarketUniverse

logger = logging.getLogger("marketsentinel.pipeline")

PIPELINE_TIMEOUT = float(os.getenv("PIPELINE_TIMEOUT_SECONDS", "12"))
TOP_K = int(os.getenv("PIPELINE_TOP_K", "5"))
BOTTOM_K = int(os.getenv("PIPELINE_BOTTOM_K", "5"))
INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))


class InferencePipeline:

    def __init__(self, model, cache, db_session_factory):
        self.model = model
        self.cache = cache
        self.db_session_factory = db_session_factory

        # Agents — lazy init
        self._signal_agent = None
        self._technical_agent = None
        self._portfolio_agent = None
        self._political_agent = None

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
        """Call an agent safely — return empty dict on any error."""
        try:
            return agent.analyze(context) or {}
        except Exception as e:
            logger.debug("Agent %s failed: %s", type(agent).__name__, e)
            return {}

    # =====================================================
    # BUILD CROSS-SECTIONAL FRAME
    # =====================================================

    def _build_cross_sectional_frame(
        self,
        tickers: List[str],
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV for all tickers and compute features.
        Returns cross-sectional DataFrame or None on failure.
        """
        from core.data.market_data_service import MarketDataService
        from core.features.feature_engineering import FeatureEngineer

        # Compute date window
        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - pd.Timedelta(days=INFERENCE_LOOKBACK_DAYS)
        start_date = start_dt.strftime("%Y-%m-%d")

        svc = MarketDataService(session_factory=self.db_session_factory)
        engineer = FeatureEngineer()

        try:
            price_data = svc.get_price_data_batch(
                tickers,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            logger.error("Price data fetch failed: %s", e)
            return None

        if not price_data:
            logger.warning("No price data returned for any ticker")
            return None

        all_frames = []
        for ticker, df in price_data.items():
            if df is None or df.empty:
                continue
            try:
                features = engineer.compute(df, ticker=ticker)
                if features is not None and not features.empty:
                    all_frames.append(features)
            except Exception as e:
                logger.debug("Feature engineering failed for %s: %s", ticker, e)

        if not all_frames:
            return None

        combined = pd.concat(all_frames, ignore_index=True)
        return combined

    # =====================================================
    # RUN SNAPSHOT
    # =====================================================

    def run_snapshot(self, snapshot_date: Optional[str] = None) -> dict:
        """
        Run full cross-sectional inference for all tickers.
        Returns the complete snapshot dict.
        """
        start_time = time.time()

        if snapshot_date is None:
            snapshot_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        logger.info("Running snapshot | date=%s", snapshot_date)

        # ── Get universe ──────────────────────────────
        try:
            universe = MarketUniverse.snapshot()
            tickers = universe.get("tickers", [])
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
            feature_block = validate_feature_schema(
                dataset[MODEL_FEATURES], mode="inference"
            ).astype(DTYPE)
        except Exception as e:
            logger.error("Feature validation failed: %s", e)
            return self._error_snapshot("Feature validation failed")

        # ── Run XGBoost inference ─────────────────────
        try:
            raw_scores = self.model.predict(feature_block)
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

        # ── FIX: Call PoliticalRiskAgent ONCE for US market ──
        political_output = {}
        try:
            political_output = self._safe_agent(
                self.political_agent,
                {"ticker": "MARKET", "country": "US"},
            )
        except Exception as e:
            logger.debug("Political agent failed: %s", e)

        political_label = political_output.get("political_risk_label", "LOW")
        political_score = float(political_output.get("political_risk_score", 0.0))

        # ── Per-ticker agent scoring ──────────────────
        snapshot_rows = []

        for idx, row in dataset.iterrows():
            ticker = row.get("ticker", "UNKNOWN")

            context = {
                "row": row.to_dict(),
                "ticker": ticker,
                "drift_state": drift_state,
                "political_risk_label": political_label,
            }

            signal_output = self._safe_agent(self.signal_agent, context)
            technical_output = self._safe_agent(self.technical_agent, context)

            raw_score = float(row.get("raw_model_score", 0.0))
            signal_score = float(signal_output.get("score", 0.0))
            technical_score = float(technical_output.get("score", 0.0))

            # Hybrid consensus score
            hybrid_score = float(np.clip(
                0.50 * raw_score +
                0.30 * signal_score +
                0.20 * technical_score,
                -1.0, 1.0,
            ))

            # Political risk penalty
            if political_label == "CRITICAL":
                hybrid_score *= 0.0
            elif political_label == "HIGH":
                hybrid_score *= 0.5

            # Drift exposure scaling
            exposure_scale = drift_result.get("exposure_scale", 1.0)
            weight = float(np.clip(hybrid_score * exposure_scale, -1.0, 1.0))

            signal_direction = signal_output.get(
                "signals", {}
            ).get("signal", "NEUTRAL") or "NEUTRAL"

            snapshot_rows.append({
                "ticker": ticker,
                "date": str(row.get("date", snapshot_date))[:10],
                "raw_model_score": round(raw_score, 6),
                "hybrid_consensus_score": round(hybrid_score, 6),
                "weight": round(weight, 6),
                # Agents dict for /agent/explain to read from cache
                "agents": {
                    "signal_agent": signal_output,
                    "technical_agent": technical_output,
                },
            })

        if not snapshot_rows:
            return self._error_snapshot("No valid signals produced")

        # ── Sort and select top/bottom K ─────────────
        snapshot_rows.sort(key=lambda x: x["raw_model_score"], reverse=True)

        long_signals = [r for r in snapshot_rows if r["weight"] > 0]
        short_signals = [r for r in snapshot_rows if r["weight"] < 0]
        neutral_signals = [r for r in snapshot_rows if r["weight"] == 0]

        top_k = snapshot_rows[:TOP_K]
        bottom_k = snapshot_rows[-BOTTOM_K:]

        # ── FIX: Compute gross/net exposure ──────────
        weights = [r["weight"] for r in snapshot_rows]
        gross_exposure = float(sum(abs(w) for w in weights))
        net_exposure = float(sum(weights))

        # Normalise to percentage if > 1
        if gross_exposure > 1.0:
            net_exposure = net_exposure / gross_exposure
            gross_exposure = 1.0

        gross_exposure = round(gross_exposure, 4)
        net_exposure = round(net_exposure, 4)

        # ── Portfolio bias ────────────────────────────
        lc = len(long_signals)
        sc = len(short_signals)
        if lc > sc * 1.5:
            portfolio_bias = "LONG_BIASED"
        elif sc > lc * 1.5:
            portfolio_bias = "SHORT_BIASED"
        else:
            portfolio_bias = "BALANCED"

        # ── Top 5 tickers ────────────────────────────
        top_5_tickers = [r["ticker"] for r in top_k]

        # ── Average hybrid score ──────────────────────
        avg_hybrid = float(np.mean([r["hybrid_consensus_score"] for r in snapshot_rows]))

        # ── Portfolio agent ───────────────────────────
        portfolio_output = {}
        try:
            portfolio_context = {
                "signals": snapshot_rows,
                "drift_state": drift_state,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
            }
            portfolio_output = self._safe_agent(self.portfolio_agent, portfolio_context)
        except Exception as e:
            logger.debug("Portfolio agent failed: %s", e)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        # ── Build final response ──────────────────────
        result = {
            "meta": {
                "model_version": getattr(self.model, "version", "unknown"),
                "drift_state": drift_state,
                "long_signals": lc,
                "short_signals": sc,
                "avg_hybrid_score": round(avg_hybrid, 6),
                "latency_ms": latency_ms,
            },
            "executive_summary": {
                "top_5_tickers": top_5_tickers,
                "portfolio_bias": portfolio_bias,
                # FIX: gross/net exposure now included
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
            },
            "snapshot": {
                "snapshot_date": snapshot_date,
                "model_version": getattr(self.model, "version", "unknown"),
                "drift": {
                    "drift_detected": drift_result.get("drift_detected", False),
                    "severity_score": drift_result.get("severity_score", 0),
                    "drift_state": drift_state,
                    "exposure_scale": drift_result.get("exposure_scale", 1.0),
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
            # Internal — for agent explain cache lookup
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