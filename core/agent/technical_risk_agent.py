# =========================================================
# TECHNICAL & RISK AGENT
# Hybrid Multi-Agent Architecture
# =========================================================

import numpy as np
from typing import Dict, Any, List

from core.agent.base_agent import BaseAgent


class TechnicalRiskAgent(BaseAgent):
    """
    Secondary Intelligence Agent.

    Evaluates:
    - Momentum confirmation
    - EMA structure
    - RSI sanity
    - Volatility regime
    - Liquidity quality

    Produces independent score (0–1).
    """

    name = "TechnicalRiskAgent"
    weight = 0.7  # Slightly lower than model agent

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    HIGH_VOL_THRESHOLD = 1.5
    LOW_LIQUIDITY_THRESHOLD = 5e5

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
        warnings: List[str] = []
        reasoning: List[str] = []

        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)
        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)
        rsi = self._safe_float(row.get("rsi"), 50.0)
        volatility_regime = self._safe_float(row.get("regime_feature"), 0.0)
        dollar_volume = self._safe_float(row.get("dollar_volume"), 0.0)

        score_components = []

        # Momentum strength
        momentum_score = np.clip(abs(momentum_z) / 3.0, 0.0, 1.0)
        score_components.append(momentum_score)

        if abs(momentum_z) < 0.5:
            warnings.append("Weak momentum")

        # EMA structure
        ema_score = np.clip(abs(ema_ratio - 1.0) * 5.0, 0.0, 1.0)
        score_components.append(ema_score)

        # RSI sanity check
        if rsi > self.RSI_OVERBOUGHT or rsi < self.RSI_OVERSOLD:
            warnings.append("RSI extreme")
            rsi_score = 0.4
        else:
            rsi_score = 1.0

        score_components.append(rsi_score)

        # Volatility penalty
        if volatility_regime > self.HIGH_VOL_THRESHOLD:
            warnings.append("High volatility regime")
            vol_score = 0.5
        else:
            vol_score = 1.0

        score_components.append(vol_score)

        # Liquidity penalty (yfinance data quality awareness)
        if dollar_volume < self.LOW_LIQUIDITY_THRESHOLD:
            warnings.append("Low liquidity")
            liquidity_score = 0.5
        else:
            liquidity_score = 1.0

        score_components.append(liquidity_score)

        final_score = float(np.clip(np.mean(score_components), 0.0, 1.0))

        return {
            "agent_name": self.name,
            "weight": self.weight,
            "score": final_score,
            "confidence": final_score,
            "signals": {},
            "warnings": sorted(set(warnings)),
            "reasoning": reasoning
        }