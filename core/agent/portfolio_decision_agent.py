# =========================================================
# PORTFOLIO DECISION AGENT v2.2
# Main Intelligence Layer
# CV Showcase Optimized | Drift-Aware | Risk-Aware
# Stability-Enhanced | Interview-Polished
# =========================================================

import numpy as np
import os
from typing import Dict, Any, List
from datetime import datetime


class PortfolioDecisionAgent:

    TOP_K = int(os.getenv("TOP_N_STOCKS", "5"))

    # -----------------------------------------------------
    # MAIN ENTRY
    # -----------------------------------------------------

    def analyze_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:

        signals = snapshot.get("signals", [])
        drift_info = snapshot.get("drift", {})

        if not signals:

            return {
                "generated_at": datetime.utcnow().isoformat(),
                "snapshot_date": snapshot.get("snapshot_date"),
                "top_5_detailed": [],
                "portfolio_findings": {
                    "status": "no_signals"
                },
                "executive_summary": "No valid trading signals generated for this snapshot."
            }

        # =================================================
        # Rank by hybrid consensus
        # =================================================

        ranked = sorted(
            signals,
            key=lambda x: x.get("hybrid_consensus_score", 0.0),
            reverse=True
        )

        top_5 = ranked[:self.TOP_K]

        detailed_selection = []
        confidence_values = []
        risk_distribution = {}
        liquidity_warnings = 0
        long_count = 0
        short_count = 0
        hybrid_scores = []

        # =================================================
        # Build Detailed Selection
        # =================================================

        for stock in top_5:

            signal_agent = stock.get("agents", {}).get("signal_agent", {})
            tech_agent = stock.get("agents", {}).get("technical_agent", {})

            confidence = float(signal_agent.get("confidence_numeric", 0.0))
            confidence_values.append(confidence)

            hybrid_score = float(stock.get("hybrid_consensus_score", 0.0))
            hybrid_scores.append(hybrid_score)

            risk_level = signal_agent.get("risk_level", "unknown")
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

            combined_warnings = list(set(
                signal_agent.get("warnings", []) +
                tech_agent.get("warnings", [])
            ))

            if any("liquidity" in w.lower() for w in combined_warnings):
                liquidity_warnings += 1

            weight = float(stock.get("weight", 0.0))

            if weight > 0:
                long_count += 1
            elif weight < 0:
                short_count += 1

            explanation = (
                f"{stock['ticker']} selected due to strong hybrid score "
                f"({hybrid_score:.2f}). "
                f"Model confidence: {confidence:.2f}, "
                f"Risk level: {risk_level}, "
                f"Volatility regime: {signal_agent.get('volatility_regime', 'unknown')}."
            )

            detailed_selection.append({
                "ticker": stock["ticker"],
                "weight": round(weight, 6),
                "hybrid_score": round(hybrid_score, 4),
                "model_score": round(float(stock.get("raw_model_score", 0.0)), 4),
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "volatility_regime": signal_agent.get("volatility_regime"),
                "technical_bias": tech_agent.get("bias"),
                "warnings": combined_warnings,
                "explanation": explanation
            })

        # =================================================
        # Portfolio-Level Intelligence
        # =================================================

        avg_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
        confidence_std = float(np.std(confidence_values)) if confidence_values else 0.0

        gross_exposure = float(snapshot.get("gross_exposure", 0.0))
        net_exposure = float(snapshot.get("net_exposure", 0.0))

        if net_exposure > 0.05:
            portfolio_bias = "long_bias"
        elif net_exposure < -0.05:
            portfolio_bias = "short_bias"
        else:
            portfolio_bias = "market_neutral"

        weights = [abs(float(s.get("weight", 0.0))) for s in top_5]

        if weights:
            max_weight = max(weights)
        else:
            max_weight = 0.0

        if max_weight > 0.18:
            concentration_risk = "high"
        elif max_weight > 0.12:
            concentration_risk = "moderate"
        else:
            concentration_risk = "low"

        drift_state = drift_info.get("drift_state", "unknown")
        severity = float(drift_info.get("severity_score", 0.0))

        if severity > 0.7:
            drift_risk = "high"
        elif severity > 0.3:
            drift_risk = "moderate"
        else:
            drift_risk = "low"

        hybrid_dispersion = float(np.std(hybrid_scores)) if hybrid_scores else 0.0

        # =================================================
        # Portfolio Findings Block
        # =================================================

        findings = {
            "average_confidence": round(avg_confidence, 4),
            "confidence_dispersion": round(confidence_std, 4),
            "hybrid_score_dispersion_top5": round(hybrid_dispersion, 4),
            "gross_exposure": round(gross_exposure, 4),
            "net_exposure": round(net_exposure, 4),
            "portfolio_bias": portfolio_bias,
            "concentration_risk": concentration_risk,
            "risk_distribution": risk_distribution,
            "liquidity_warnings_in_top5": liquidity_warnings,
            "long_positions_in_top5": long_count,
            "short_positions_in_top5": short_count,
            "drift_state": drift_state,
            "drift_severity": round(severity, 4),
            "drift_risk_level": drift_risk
        }

        # =================================================
        # Executive Summary
        # =================================================

        executive_summary = (
            f"Hybrid AI consensus selected top {len(top_5)} equities with "
            f"average confidence {avg_confidence:.2f}. "
            f"Portfolio bias: {portfolio_bias.replace('_', ' ')}. "
            f"Drift condition: {drift_state} (risk: {drift_risk}). "
            f"Gross exposure: {gross_exposure:.2f}, "
            f"Net exposure: {net_exposure:.2f}. "
            f"Concentration risk assessed as {concentration_risk}. "
            f"Signal dispersion among top selections: {hybrid_dispersion:.2f}."
        )

        # =================================================
        # Final Structured Output
        # =================================================

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "model_version": snapshot.get("model_version"),
            "snapshot_date": snapshot.get("snapshot_date"),
            "top_5_detailed": detailed_selection,
            "portfolio_findings": findings,
            "executive_summary": executive_summary
        }