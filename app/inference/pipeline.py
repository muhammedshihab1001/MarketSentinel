# =========================================================
# INSTITUTIONAL INFERENCE PIPELINE v5.9.1
#
# FIX v5.8: run_snapshot() was processing ALL 27,400 rows
#   (274 days × 100 tickers) in the per-ticker loop.
#   Fix: filter to latest date per ticker before scoring.
#   Result: 100 rows → ~5s. signals=100.
#
# NEW v5.9: Added top_5_rationale to snapshot response.
#   Each approved stock now includes:
#     - which agents approved it and their scores
#     - why it was selected (natural language reason)
#     - what signal direction and confidence each agent gave
#     - sector, drift context, political risk context
#   Exposed in executive_summary.top_5_rationale array.
#   Used by frontend Agent page to show approved stock cards.
#
# FIX v5.9.1:
#   - Double "volatility" word bug fixed in selection_reason.
#     "low_volatility" was producing "low volatility volatility
#     regime" because replace('_', ' ') kept the word.
#     Fixed with _vol_label() mapping: strips suffix correctly.
#   - Duplicate STORE_PREDICTIONS block removed.
#     Prediction storage was running twice per snapshot,
#     causing unnecessary DB writes and log noise.
# =========================================================

import time
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from core.schema.feature_schema import MODEL_FEATURES, validate_feature_schema, DTYPE
from core.market.universe import MarketUniverse

logger = logging.getLogger("marketsentinel.pipeline")

PIPELINE_TIMEOUT = float(os.getenv("PIPELINE_TIMEOUT_SECONDS", "12"))
TOP_K = int(os.getenv("PIPELINE_TOP_K", "5"))
BOTTOM_K = int(os.getenv("PIPELINE_BOTTOM_K", "5"))
INFERENCE_LOOKBACK_DAYS = int(os.getenv("INFERENCE_LOOKBACK_DAYS", "400"))

# =========================================================
# FIX v5.9.1 — Volatility regime display mapping
# Converts internal enum to clean human-readable label.
# Prevents double-word: "low_volatility" → "low"
# Then used as: "low volatility regime" ✅
# Not:          "low volatility volatility regime" ❌
# =========================================================

_VOL_DISPLAY: Dict[str, str] = {
    "high_volatility": "high",
    "low_volatility":  "low",
    "normal":          "normal",
}


def _vol_label(regime: str) -> str:
    """Convert volatility_regime to display word without duplication."""
    return _VOL_DISPLAY.get(regime, regime.replace("_", " "))


class InferencePipeline:

    def __init__(self, model=None, cache=None, db_session_factory=None):
        self._model = model
        self._cache = cache
        self._db_session_factory = db_session_factory

        self._signal_agent = None
        self._technical_agent = None
        self._portfolio_agent = None
        self._political_agent = None

    # =====================================================
    # MODEL ACCESSOR
    # =====================================================

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from app.api.routes.predict import _model_loader
            if _model_loader is not None:
                return _model_loader
        except ImportError:
            pass
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

        svc = MarketDataService()
        engineer = FeatureEngineer()

        try:
            price_result = svc.get_price_data_batch(
                tickers,
                start_date=start_date,
                end_date=end_date,
            )
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

        all_frames = []
        for ticker, df in price_data.items():
            if df is None or df.empty:
                continue
            df = df.copy()
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            all_frames.append(df)

        if not all_frames:
            return None

        combined_prices = pd.concat(all_frames, ignore_index=True)

        try:
            features = engineer.build_feature_pipeline(combined_prices, training=False)
        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            return None

        return features

    # =====================================================
    # FILTER TO LATEST DATE PER TICKER (v5.8 fix)
    # =====================================================

    @staticmethod
    def _filter_latest_per_ticker(dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset is None or dataset.empty:
            return dataset

        dataset = dataset.copy()
        dataset["date"] = pd.to_datetime(dataset["date"], utc=True, errors="coerce")
        dataset = dataset.dropna(subset=["date"])

        latest = (
            dataset
            .sort_values("date")
            .groupby("ticker", sort=False)
            .tail(1)
            .reset_index(drop=True)
        )

        logger.info(
            "Latest-per-ticker filter | input_rows=%d output_rows=%d tickers=%d",
            len(dataset), len(latest), latest["ticker"].nunique(),
        )

        return latest

    # =====================================================
    # BUILD TOP-5 RATIONALE (v5.9)
    # =====================================================

    @staticmethod
    def _build_top5_rationale(
        top5_rows: List[Dict],
        drift_state: str,
        political_label: str,
        political_score: float,
    ) -> List[Dict]:
        """
        Build detailed rationale for each top-5 approved stock.

        Args:
            top5_rows:        list of snapshot_rows for top-5 tickers
            drift_state:      current drift state string
            political_label:  political risk label (LOW/MEDIUM/HIGH/CRITICAL)
            political_score:  political risk score 0.0-1.0

        Returns:
            List of rationale dicts, one per stock.
        """
        rationale_list = []

        for rank, row in enumerate(top5_rows, start=1):
            ticker = row.get("ticker", "")
            raw_score = row.get("raw_model_score", 0.0)
            hybrid_score = row.get("hybrid_consensus_score", 0.0)
            weight = row.get("weight", 0.0)

            agents = row.get("agents", {})
            signal_out = agents.get("signal_agent", {}) or {}
            technical_out = agents.get("technical_agent", {}) or {}

            # ── Signal agent fields ───────────────────────────
            signal_direction = (
                signal_out.get("signals", {}).get("signal")
                or signal_out.get("signal")
                or ("LONG" if weight > 0.01 else ("SHORT" if weight < -0.01 else "NEUTRAL"))
            )
            confidence = float(signal_out.get("confidence_numeric", 0.0))
            risk_level = signal_out.get("risk_level", "unknown")
            governance_score = signal_out.get("governance_score", 0)
            signal_warnings = signal_out.get("warnings", []) or []
            signal_agent_score = float(signal_out.get("score", 0.0))

            # ── Technical agent fields ────────────────────────
            volatility_regime = (
                technical_out.get("signals", {}).get("volatility_regime")
                or technical_out.get("volatility_regime")
                or "normal"
            )
            technical_bias = (
                technical_out.get("bias")
                or technical_out.get("signals", {}).get("bias")
                or "neutral"
            )
            technical_agent_score = float(technical_out.get("score", 0.0))
            technical_warnings = technical_out.get("warnings", []) or []

            # ── Approval status per agent ─────────────────────
            signal_approved = signal_agent_score > 0.3 and risk_level != "high"
            technical_approved = technical_agent_score > 0.3
            political_approved = political_label not in ("HIGH", "CRITICAL")

            agents_approved = []
            agents_flagged = []

            if signal_approved:
                agents_approved.append("SignalAgent")
            else:
                agents_flagged.append(
                    f"SignalAgent (score={signal_agent_score:.2f}, risk={risk_level})"
                )

            if technical_approved:
                agents_approved.append("TechnicalRiskAgent")
            else:
                agents_flagged.append(
                    f"TechnicalRiskAgent (score={technical_agent_score:.2f})"
                )

            if political_approved:
                agents_approved.append("PoliticalRiskAgent")
            else:
                agents_flagged.append(
                    f"PoliticalRiskAgent (label={political_label})"
                )

            # ── Natural language selection reason ─────────────
            # FIX v5.9.1: _vol_label() strips "_volatility" suffix
            # "low_volatility" → "low" → "low volatility regime" ✅
            # Previously: .replace('_', ' ') → "low volatility"
            # → "low volatility volatility regime" ❌
            vol_display = _vol_label(volatility_regime)

            reason_parts = []

            reason_parts.append(
                f"{ticker} ranked #{rank} with hybrid consensus score "
                f"{hybrid_score:.4f} (raw model: {raw_score:.4f})."
            )

            reason_parts.append(
                f"Signal: {signal_direction} | Confidence: {confidence:.1%} | "
                f"Risk level: {risk_level}."
            )

            reason_parts.append(
                f"Technical bias is {technical_bias} with "
                f"{vol_display} volatility regime."
            )

            if drift_state not in ("none", "low"):
                reason_parts.append(
                    f"Drift state is {drift_state} — position weight scaled down "
                    f"by exposure_scale to manage regime risk."
                )
            else:
                reason_parts.append("Model drift is within normal range.")

            if political_label in ("HIGH", "CRITICAL"):
                reason_parts.append(
                    f"Political risk is {political_label} "
                    f"(score: {political_score:.2f}) — "
                    f"hybrid score reduced by political risk overlay."
                )
            else:
                reason_parts.append(
                    f"Political environment is LOW risk "
                    f"(score: {political_score:.2f})."
                )

            if signal_warnings or technical_warnings:
                all_warnings = list(set(signal_warnings + technical_warnings))
                reason_parts.append(
                    f"Active warnings: {', '.join(all_warnings[:3])}."
                )

            reason_parts.append(
                f"Portfolio weight assigned: {weight:.4f} "
                f"({abs(weight) * 100:.2f}% of portfolio)."
            )

            rationale_list.append({
                "rank": rank,
                "ticker": ticker,
                "signal": signal_direction,
                "hybrid_score": round(hybrid_score, 6),
                "raw_model_score": round(raw_score, 6),
                "weight": round(weight, 6),
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "governance_score": governance_score,
                "volatility_regime": volatility_regime,
                "technical_bias": technical_bias,
                "drift_context": drift_state,
                "political_context": political_label,
                "agent_scores": {
                    "signal_agent": round(signal_agent_score, 4),
                    "technical_agent": round(technical_agent_score, 4),
                    "raw_model": round(raw_score, 4),
                },
                "agents_approved": agents_approved,
                "agents_flagged": agents_flagged,
                "warnings": list(set(signal_warnings + technical_warnings)),
                "selection_reason": " ".join(reason_parts),
            })

        return rationale_list

    # =====================================================
    # RUN SNAPSHOT
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

        # ── Build feature frame (full lookback) ───────
        dataset = self._build_cross_sectional_frame(tickers, snapshot_date)

        if dataset is None or dataset.empty:
            return self._error_snapshot("Feature engineering failed for all tickers")

        # ── Filter to ONE row per ticker (v5.8 fix) ───
        dataset = self._filter_latest_per_ticker(dataset)

        if dataset.empty:
            return self._error_snapshot("No latest-date rows found after filter")

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

        # ── Political risk ────────────────────────────
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

        # ── Per-ticker scoring loop ───────────────────
        snapshot_rows = []
        tickers_set = set(tickers)
        exposure_scale = float(drift_result.get("exposure_scale", 1.0))

        for idx, row in dataset.iterrows():
            ticker = row.get("ticker", "UNKNOWN")

            if ticker not in tickers_set:
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

        # ── Portfolio agent ───────────────────────────
        portfolio_output = {}
        try:
            portfolio_context = {
                "signals": snapshot_rows,
                "drift": drift_result,
                "drift_state": drift_state,
                "gross_exposure": round(gross_exposure, 4),
                "net_exposure": round(net_exposure, 4),
            }
            portfolio_output = self._safe_agent(self.portfolio_agent, portfolio_context)
        except Exception as e:
            logger.debug("Portfolio agent failed: %s", e)

        # ── NEW v5.9: Build top-5 rationale ──────────
        top_5_rationale = self._build_top5_rationale(
            top5_rows=top_5,
            drift_state=drift_state,
            political_label=political_label,
            political_score=political_score,
        )

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
                "top_5_rationale": top_5_rationale,
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

        # ── FIX v5.9.1: Single prediction storage block ──────
        # Removed duplicate — was running twice per snapshot.
        # STORE_PREDICTIONS=1 enables IC stats computation.
        store_preds = os.getenv("STORE_PREDICTIONS", "0") == "1"
        if store_preds:
            try:
                from core.db.repository import PredictionRepository
                model_ver = getattr(loader, "version", "unknown")
                pred_records = [
                    {
                        "ticker": r["ticker"],
                        "date": r["date"],
                        "model_version": model_ver,
                        "raw_model_score": r["raw_model_score"],
                        "hybrid_score": r["hybrid_consensus_score"],
                        "weight": r["weight"],
                        "signal": (
                            "LONG" if r["weight"] > 0.01
                            else "SHORT" if r["weight"] < -0.01
                            else "NEUTRAL"
                        ),
                        "drift_state": drift_state,
                    }
                    for r in snapshot_rows
                ]
                PredictionRepository.store_predictions(pred_records)
                logger.info(
                    "Predictions stored | date=%s | count=%d",
                    snapshot_date, len(pred_records),
                )
            except Exception as e:
                logger.warning("Prediction storage failed (non-blocking): %s", e)

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
                "top_5_rationale": [],
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
