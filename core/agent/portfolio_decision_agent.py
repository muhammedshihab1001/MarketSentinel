"""
MarketSentinel v4.1.0

Portfolio Decision Agent — aggregates per-ticker agent outputs into a
ranked portfolio selection with portfolio-level risk analysis.

Responsibilities:
    - Rank stocks by hybrid consensus score
    - Build detailed per-ticker selection block
    - Compute portfolio-level metrics (bias, concentration, drift)
    - Produce executive summary for API / dashboard consumption

Inherits BaseAgent so it participates in the unified agent architecture.
Entry point: analyze(context) — wraps analyze_snapshot() for compatibility.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from core.agent.base_agent import BaseAgent


class PortfolioDecisionAgent(BaseAgent):
    """
    Final decision layer — aggregates SignalAgent + TechnicalRiskAgent
    outputs into a ranked portfolio selection.

    Uses analyze(context) as the BaseAgent-compliant entry point.
    context must contain a "snapshot" key with the full signal snapshot.
    """

    name        = "PortfolioDecisionAgent"
    weight      = 1.0
    description = (
        "Ranks stocks by hybrid consensus score and produces "
        "portfolio-level risk analysis and executive summary."
    )

    TOP_K = int(os.getenv("TOP_N_STOCKS", "5"))

    # ─────────────────────────────────────────────────────────────────────────
    # BaseAgent CONTRACT — required abstract method implementation
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        BaseAgent-compliant entry point.

        Expects context to contain either:
            {"snapshot": <full snapshot dict>}   (preferred)
            {flat snapshot dict}                  (legacy fallback)
        """
        snapshot = context.get("snapshot", context)
        return self.analyze_snapshot(snapshot)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN PORTFOLIO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full portfolio decision pipeline.

        Parameters
        ----------
        snapshot : dict containing:
            signals        : list of per-ticker signal dicts
            drift          : drift detector output dict
            gross_exposure : float
            net_exposure   : float
            snapshot_date  : ISO date string
            model_version  : str
        """
        signals    = snapshot.get("signals", [])
        drift_info = snapshot.get("drift", {})
        generated_at = datetime.now(timezone.utc).isoformat()

        # ── Empty signal guard ────────────────────────────────────────────────
        if not signals:
            return {
                "generated_at":      generated_at,
                "snapshot_date":     snapshot.get("snapshot_date"),
                "top_selections":    [],
                "portfolio_findings": {"status": "no_signals"},
                "executive_summary": (
                    "No valid trading signals generated for this snapshot."
                ),
            }

        # ── Rank by hybrid consensus score ────────────────────────────────────
        ranked = sorted(
            signals,
            key=lambda x: self._safe_float(x.get("hybrid_consensus_score"), 0.0),
            reverse=True,
        )
        top_k = ranked[: self.TOP_K]

        # ── Build detailed selection ──────────────────────────────────────────
        detailed_selection  = []
        confidence_values:  List[float] = []
        hybrid_scores:      List[float] = []
        risk_distribution:  Dict[str, int] = {}
        liquidity_warnings  = 0
        long_count          = 0
        short_count         = 0

        for stock in top_k:
            signal_agent = stock.get("agents", {}).get("signal_agent", {})
            tech_agent   = stock.get("agents", {}).get("technical_agent", {})

            confidence   = self._safe_float(signal_agent.get("confidence_numeric"), 0.0)
            hybrid_score = self._safe_float(stock.get("hybrid_consensus_score"),    0.0)
            weight       = self._safe_float(stock.get("weight"),                    0.0)

            confidence_values.append(confidence)
            hybrid_scores.append(hybrid_score)

            risk_level = signal_agent.get("risk_level", "unknown")
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

            # Merge warnings — preserve insertion order, deduplicate
            sa_warnings   = signal_agent.get("warnings", [])
            tech_warnings = tech_agent.get("warnings", [])
            combined_warnings = list(
                dict.fromkeys(sa_warnings + tech_warnings)
            )

            if any("liquidity" in w.lower() for w in combined_warnings):
                liquidity_warnings += 1

            # ── Direction from signal field — not weight sign ─────────────────
            direction = (
                signal_agent.get("signals", {}).get("direction")
                or signal_agent.get("signal")
                or "NEUTRAL"
            )
            if direction == "LONG":
                long_count  += 1
            elif direction == "SHORT":
                short_count += 1

            explanation = (
                f"{stock.get('ticker', '?')} selected with hybrid score "
                f"{hybrid_score:.2f}. "
                f"Direction: {direction}. "
                f"Model confidence: {confidence:.2f}. "
                f"Risk level: {risk_level}. "
                f"Volatility regime: "
                f"{signal_agent.get('volatility_regime', 'unknown')}."
            )

            detailed_selection.append({
                "ticker":            stock.get("ticker"),
                "weight":            round(weight, 6),
                "direction":         direction,
                "hybrid_score":      round(hybrid_score, 4),
                "model_score":       round(self._safe_float(stock.get("raw_model_score")), 4),
                "confidence":        round(confidence, 4),
                "risk_level":        risk_level,
                "volatility_regime": signal_agent.get("volatility_regime"),
                "technical_bias":    tech_agent.get("bias"),
                "warnings":          combined_warnings,
                "explanation":       explanation,
            })

        # ── Portfolio-level metrics ───────────────────────────────────────────
        avg_confidence   = float(np.mean(confidence_values)) if confidence_values else 0.0
        confidence_std   = float(np.std(confidence_values))  if confidence_values else 0.0
        hybrid_dispersion = float(np.std(hybrid_scores))     if hybrid_scores     else 0.0

        gross_exposure = self._safe_float(snapshot.get("gross_exposure"), 0.0)
        net_exposure   = self._safe_float(snapshot.get("net_exposure"),   0.0)

        # ── Portfolio bias ────────────────────────────────────────────────────
        if net_exposure > 0.05:
            portfolio_bias = "long_bias"
        elif net_exposure < -0.05:
            portfolio_bias = "short_bias"
        else:
            portfolio_bias = "market_neutral"

        # ── Concentration risk ────────────────────────────────────────────────
        abs_weights = [abs(self._safe_float(s.get("weight"), 0.0)) for s in top_k]
        max_weight  = max(abs_weights) if abs_weights else 0.0

        if max_weight > 0.18:
            concentration_risk = "high"
        elif max_weight > 0.12:
            concentration_risk = "moderate"
        else:
            concentration_risk = "low"

        # ── Drift risk ────────────────────────────────────────────────────────
        drift_state = drift_info.get("drift_state", "unknown")
        severity    = self._safe_float(drift_info.get("severity_score"), 0.0)

        if severity > 0.7:
            drift_risk = "high"
        elif severity > 0.3:
            drift_risk = "moderate"
        else:
            drift_risk = "low"

        # ── Portfolio findings ────────────────────────────────────────────────
        findings = {
            "average_confidence":           round(avg_confidence,    4),
            "confidence_dispersion":        round(confidence_std,    4),
            "hybrid_score_dispersion_topk": round(hybrid_dispersion, 4),
            "gross_exposure":               round(gross_exposure,    4),
            "net_exposure":                 round(net_exposure,      4),
            "portfolio_bias":               portfolio_bias,
            "concentration_risk":           concentration_risk,
            "risk_distribution":            risk_distribution,
            "liquidity_warnings_in_topk":   liquidity_warnings,
            "long_positions_in_topk":       long_count,
            "short_positions_in_topk":      short_count,
            "drift_state":                  drift_state,
            "drift_severity":               round(severity, 4),
            "drift_risk_level":             drift_risk,
        }

        # ── Executive summary ─────────────────────────────────────────────────
        executive_summary = (
            f"Hybrid AI consensus selected top {len(top_k)} equities with "
            f"average confidence {avg_confidence:.2f}. "
            f"Portfolio bias: {portfolio_bias.replace('_', ' ')}. "
            f"Drift condition: {drift_state} (risk: {drift_risk}). "
            f"Gross exposure: {gross_exposure:.2f}, "
            f"net exposure: {net_exposure:.2f}. "
            f"Concentration risk: {concentration_risk}. "
            f"Signal dispersion among top selections: {hybrid_dispersion:.2f}."
        )

        # ── Final structured output ───────────────────────────────────────────
        return {
            "generated_at":       generated_at,
            "model_version":      snapshot.get("model_version"),
            "snapshot_date":      snapshot.get("snapshot_date"),
            "top_selections":     detailed_selection,       # key no longer hardcodes "5"
            "portfolio_findings": findings,
            "executive_summary":  executive_summary,
        }