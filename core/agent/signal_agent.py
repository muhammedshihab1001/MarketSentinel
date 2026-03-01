# =========================================================
# INSTITUTIONAL SIGNAL AGENT v2 (Multi-Agent Ready)
# =========================================================

import numpy as np
from typing import Dict, Any, List, Optional

EPSILON = 1e-12


class SignalAgent:

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    EMA_BULLISH = 1.05
    EMA_BEARISH = 0.95

    MOMENTUM_STRONG = 1.0
    DISPERSION_WEAK = 0.05

    Z_VERY_STRONG = 2.0
    Z_STRONG = 1.25
    Z_MODERATE = 0.75

    MAX_SCORE_SANITY = 10.0

    MAX_POSITION_SIZE = 1.0
    MIN_POSITION_SIZE = 0.0

    MIN_CONFIDENCE_TO_TRADE = 0.30
    HIGH_RISK_CONFIDENCE_THRESHOLD = 0.45

    # ---------------------------------------------------------

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

    # ---------------------------------------------------------

    def _alignment_score(self, signal, momentum_z, ema_ratio):
        alignment = 0

        if signal == "LONG":
            if momentum_z > 0:
                alignment += 1
            if ema_ratio > self.EMA_BULLISH:
                alignment += 1

        if signal == "SHORT":
            if momentum_z < 0:
                alignment += 1
            if ema_ratio < self.EMA_BEARISH:
                alignment += 1

        return alignment

    # ---------------------------------------------------------

    def _confidence_numeric(self, abs_score: float) -> float:
        return float(np.clip(abs_score / self.Z_VERY_STRONG, 0.0, 1.0))

    def _volatility_adjusted_confidence(
        self,
        confidence_numeric: float,
        volatility: float,
    ) -> float:

        if volatility <= 0:
            return confidence_numeric

        penalty = np.clip(volatility / 2.0, 0.0, 1.0)
        adjusted = confidence_numeric * (1 - 0.3 * penalty)

        return float(np.clip(adjusted, 0.0, 1.0))

    # ---------------------------------------------------------

    def _suggest_position_size(
        self,
        confidence_numeric: float,
        risk_level: str,
        volatility_regime: str,
    ) -> float:

        base = confidence_numeric

        if risk_level == "elevated":
            base *= 0.6
        elif risk_level == "high":
            base *= 0.75

        if volatility_regime == "high_volatility":
            base *= 0.7

        return float(np.clip(base, self.MIN_POSITION_SIZE, self.MAX_POSITION_SIZE))

    # ---------------------------------------------------------
    # MAIN ANALYSIS
    # ---------------------------------------------------------

    def analyze(
        self,
        row: Dict[str, Any],
        probability_stats: Optional[Dict[str, float]] = None,
        drift_score: Optional[float] = None,
    ) -> Dict[str, Any]:

        raw_model_score = self._safe_float(
            row.get("raw_model_score", row.get("alpha_score", row.get("score"))),
            0.0
        )

        final_score = self._safe_float(row.get("score"), 0.0)
        final_score = float(np.clip(final_score, -self.MAX_SCORE_SANITY, self.MAX_SCORE_SANITY))

        signal = row.get("signal", "NEUTRAL")
        if signal not in {"LONG", "SHORT", "NEUTRAL"}:
            signal = "NEUTRAL"

        volatility = self._safe_float(row.get("volatility"), 0.0)
        rsi = self._safe_float(row.get("rsi"), 50.0)
        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)
        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)
        regime_feature = self._safe_float(row.get("regime_feature"), 0.0)
        breadth = self._safe_float(row.get("breadth"), 0.5)

        warnings: List[str] = []
        reasoning: List[str] = []

        abs_score = abs(final_score)

        # =====================================================
        # ALPHA COMPONENT
        # =====================================================

        alpha_strength = float(abs_score)
        confidence_numeric = self._confidence_numeric(abs_score)
        confidence_numeric = self._volatility_adjusted_confidence(
            confidence_numeric,
            volatility
        )

        # =====================================================
        # TECHNICAL COMPONENT
        # =====================================================

        alignment = self._alignment_score(signal, momentum_z, ema_ratio)
        technical_score = alignment / 2.0  # normalize to 0–1

        # =====================================================
        # RISK COMPONENT
        # =====================================================

        if regime_feature > 1.5:
            volatility_regime = "high_volatility"
        else:
            volatility_regime = "normal"

        if volatility_regime == "high_volatility":
            risk_level = "elevated"
        elif abs_score < 0.5:
            risk_level = "high"
        elif abs_score < 1.0:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # =====================================================
        # DRIFT ADJUSTMENT
        # =====================================================

        drift_flag = False
        if drift_score is not None:
            drift_score = self._safe_float(drift_score, 0.0)
            if drift_score > 0.5:
                drift_flag = True
                confidence_numeric *= 0.7
                confidence_numeric = float(np.clip(confidence_numeric, 0.0, 1.0))

        # =====================================================
        # COMPOSITE AGENT SCORE (For Orchestrator)
        # =====================================================

        agent_score = float(np.clip(
            0.5 * confidence_numeric +
            0.3 * technical_score +
            0.2 * (0 if drift_flag else 1),
            0.0,
            1.0
        ))

        # =====================================================
        # TRADE APPROVAL
        # =====================================================

        trade_approved = (
            signal != "NEUTRAL"
            and confidence_numeric > self.MIN_CONFIDENCE_TO_TRADE
            and not drift_flag
        )

        position_size_hint = self._suggest_position_size(
            confidence_numeric,
            risk_level,
            volatility_regime
        )

        # =====================================================
        # GOVERNANCE SCORE
        # =====================================================

        governance_score = int(
            np.clip(
                agent_score * 100,
                0,
                100
            )
        )

        explanation = (
            f"{signal} | raw={raw_model_score:.2f} | "
            f"final={final_score:.2f} | "
            f"agent_score={agent_score:.2f} | "
            f"risk={risk_level}"
        )

        return {
            "signal": signal,
            "alpha_strength": alpha_strength,
            "confidence_numeric": confidence_numeric,
            "technical_score": technical_score,
            "agent_score": agent_score,
            "risk_level": risk_level,
            "volatility_regime": volatility_regime,
            "alignment_score": alignment,
            "position_size_hint": position_size_hint,
            "trade_approved": trade_approved,
            "drift_flag": drift_flag,
            "governance_score": governance_score,
            "reasoning": sorted(set(reasoning)),
            "warnings": sorted(set(warnings)),
            "explanation": explanation,
        }