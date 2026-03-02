# =========================================================
# TECHNICAL & RISK AGENT v2.0
# Hybrid Multi-Agent Architecture
# CV-Optimized | Lightweight | yfinance-aware
# =========================================================

import numpy as np
from typing import Dict, Any, List

from core.agent.base_agent import BaseAgent


class TechnicalRiskAgent(BaseAgent):
    """
    Secondary Intelligence Agent.

    Evaluates:
    - Momentum strength
    - EMA structure alignment
    - RSI sanity
    - Volatility regime
    - Liquidity quality (important for yfinance data)

    Produces independent score (0–1).
    """

    name = "TechnicalRiskAgent"
    weight = 0.7  # Slightly lower than model agent

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    HIGH_VOL_THRESHOLD = 1.5
    LOW_LIQUIDITY_THRESHOLD = 5e5

    MOMENTUM_STRONG = 1.0

    # -----------------------------------------------------

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
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

        warnings: List[str] = []
        reasoning: List[str] = []

        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)
        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)
        rsi = self._safe_float(row.get("rsi"), 50.0)
        regime_feature = self._safe_float(row.get("regime_feature"), 0.0)
        dollar_volume = self._safe_float(row.get("dollar_volume"), 0.0)

        score_components = []

        # -------------------------------------------------
        # Momentum Strength
        # -------------------------------------------------

        momentum_strength = np.clip(abs(momentum_z) / 3.0, 0.0, 1.0)
        score_components.append(momentum_strength)

        if abs(momentum_z) < 0.5:
            warnings.append("Weak momentum")

        # Directional bias
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
        score_components.append(ema_score)

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

        score_components.append(rsi_score)

        # -------------------------------------------------
        # Volatility Regime
        # -------------------------------------------------

        if regime_feature > self.HIGH_VOL_THRESHOLD:
            warnings.append("High volatility regime")
            vol_score = 0.5
        else:
            vol_score = 1.0

        score_components.append(vol_score)

        # -------------------------------------------------
        # Liquidity (Important for yfinance noise)
        # -------------------------------------------------

        if dollar_volume < self.LOW_LIQUIDITY_THRESHOLD:
            warnings.append("Low liquidity (data reliability risk)")
            liquidity_score = 0.5
        else:
            liquidity_score = 1.0

        score_components.append(liquidity_score)

        # -------------------------------------------------
        # Final Score
        # -------------------------------------------------

        final_score = float(np.clip(np.mean(score_components), 0.0, 1.0))

        confidence = final_score

        return {
            "agent_name": self.name,
            "weight": self.weight,
            "score": final_score,
            "confidence": confidence,
            "bias": bias,  # NEW (required by pipeline + LLM)
            "component_scores": {
                "momentum": momentum_strength,
                "ema": ema_score,
                "rsi": rsi_score,
                "volatility": vol_score,
                "liquidity": liquidity_score
            },
            "warnings": sorted(set(warnings)),
            "reasoning": reasoning
        }