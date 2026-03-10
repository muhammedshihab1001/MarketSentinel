# =========================================================
# TECHNICAL & RISK AGENT v2.2
# Hybrid Multi-Agent Architecture
# Drift-Aware | Regime-Aware | CV-Optimized
# =========================================================

import numpy as np
from typing import Dict, Any, List

from core.agent.base_agent import BaseAgent


class TechnicalRiskAgent(BaseAgent):

    name = "TechnicalRiskAgent"
    weight = 0.7

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    HIGH_VOL_THRESHOLD = 1.5
    LOW_VOL_THRESHOLD = -0.5

    LOW_LIQUIDITY_THRESHOLD = 5e5

    MOMENTUM_STRONG = 1.0

    # -----------------------------------------------------

    @staticmethod
    def _safe_float(value, default=0.0):

        try:

            if value is None:
                return default

            val = float(value)

            if not np.isfinite(val):
                return default

            return val

        except Exception:
            return default

    # -----------------------------------------------------

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:

        row = context.get("row", {})
        signal = row.get("signal", "NEUTRAL")
        drift_state = context.get("drift_state")

        warnings: List[str] = []
        reasoning: List[str] = []

        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)
        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)
        rsi = self._safe_float(row.get("rsi"), 50.0)
        regime_feature = self._safe_float(row.get("regime_feature"), 0.0)
        dollar_volume = self._safe_float(row.get("dollar_volume"), 0.0)

        # -------------------------------------------------
        # Momentum Strength
        # -------------------------------------------------

        momentum_strength = np.clip(abs(momentum_z) / 3.0, 0.0, 1.0)

        if abs(momentum_z) < 0.5:
            warnings.append("Low momentum (stable asset)")

        if momentum_z > self.MOMENTUM_STRONG:
            bias = "bullish"
        elif momentum_z < -self.MOMENTUM_STRONG:
            bias = "bearish"
        else:
            bias = "neutral"

        # -------------------------------------------------
        # EMA Structure
        # -------------------------------------------------

        ema_score = np.clip(abs(ema_ratio - 1.0) * 5.0, 0.0, 1.0)

        if signal == "LONG" and ema_ratio < 1.0:
            warnings.append("EMA not supportive of long")

        if signal == "SHORT" and ema_ratio > 1.0:
            warnings.append("EMA not supportive of short")

        # -------------------------------------------------
        # RSI Sanity
        # -------------------------------------------------

        if rsi > self.RSI_OVERBOUGHT:
            warnings.append("RSI overbought")
            rsi_score = 0.4

        elif rsi < self.RSI_OVERSOLD:
            warnings.append("RSI oversold")
            rsi_score = 0.4

        else:
            rsi_score = 1.0

        # -------------------------------------------------
        # Volatility Regime
        # -------------------------------------------------

        if regime_feature > self.HIGH_VOL_THRESHOLD:

            warnings.append("High volatility regime")

            vol_multiplier = 0.7

        elif regime_feature < self.LOW_VOL_THRESHOLD:

            reasoning.append("Low volatility stability regime")

            vol_multiplier = 1.1

        else:

            vol_multiplier = 1.0

        # -------------------------------------------------
        # Liquidity
        # -------------------------------------------------

        if dollar_volume < self.LOW_LIQUIDITY_THRESHOLD:

            warnings.append("Low liquidity (data reliability risk)")

            liquidity_score = 0.5

        else:

            liquidity_score = 1.0

        # -------------------------------------------------
        # Drift Penalty
        # -------------------------------------------------

        drift_penalty = 1.0

        if drift_state == "soft":

            drift_penalty = 0.85

            warnings.append("Soft drift environment")

        elif drift_state == "hard":

            drift_penalty = 0.70

            warnings.append("Hard drift regime")

        # -------------------------------------------------
        # SIGNAL SCORE (pure signal)
        # -------------------------------------------------

        signal_score = float(np.mean([
            momentum_strength,
            ema_score,
            rsi_score
        ]))

        # -------------------------------------------------
        # GATE MULTIPLIER (risk controls)
        # -------------------------------------------------

        gate_multiplier = (
            liquidity_score *
            drift_penalty *
            vol_multiplier
        )

        final_score = float(
            np.clip(signal_score * gate_multiplier, 0.0, 1.0)
        )

        confidence = final_score

        governance_score = int(np.clip(final_score * 100, 0, 100))

        explanation = (
            f"tech_score={final_score:.2f} | "
            f"bias={bias} | "
            f"drift={drift_state or 'none'}"
        )

        return {
            "agent_name": self.name,
            "weight": self.weight,
            "score": final_score,
            "confidence": confidence,
            "bias": bias,
            "governance_score": governance_score,
            "component_scores": {
                "momentum": momentum_strength,
                "ema": ema_score,
                "rsi": rsi_score,
                "signal_score": signal_score,
                "liquidity": liquidity_score,
                "drift_penalty": drift_penalty,
                "volatility_multiplier": vol_multiplier
            },
            "warnings": sorted(set(warnings)),
            "reasoning": reasoning,
            "explanation": explanation
        }