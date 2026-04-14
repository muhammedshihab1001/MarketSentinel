"""
MarketSentinel v4.2.1

Signal Agent — interprets XGBoost model output into a structured
trading signal with confidence, risk level, and position sizing hint.

Portfolio-safe version for noisy market data (yfinance compatible).
"""

import numpy as np
from typing import Any, Dict, List

from core.agent.base_agent import BaseAgent

EPSILON = 1e-12


class SignalAgent(BaseAgent):

    name = "SignalAgent"

    weight = 1.0

    description = (
        "Interprets XGBoost model scores and technical indicators "
        "into directional signals with confidence and position sizing."
    )

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    EMA_BULLISH = 1.05
    EMA_BEARISH = 0.95

    Z_VERY_STRONG = 2.0

    MAX_SCORE_SANITY = 10.0

    MAX_POSITION_SIZE = 1.0
    MIN_POSITION_SIZE = 0.0

    MIN_CONFIDENCE_TO_TRADE = 0.30
    LOW_VOL_CONFIDENCE = 0.20

    # ============================================================
    # SAFE FLOAT
    # ============================================================

    def _safe_float(self, value, default=0.0):

        try:
            v = float(value)

            if not np.isfinite(v):
                return default

            return v

        except Exception:
            return default

    # ============================================================
    # TECHNICAL ALIGNMENT
    # ============================================================

    def _alignment_score(self, signal, momentum_z, ema_ratio):

        alignment = 0

        if signal == "LONG":

            if momentum_z > 0:
                alignment += 1

            if ema_ratio > self.EMA_BULLISH:
                alignment += 1

        elif signal == "SHORT":

            if momentum_z < 0:
                alignment += 1

            if ema_ratio < self.EMA_BEARISH:
                alignment += 1

        return alignment

    # ============================================================
    # CONFIDENCE
    # ============================================================

    def _confidence_numeric(self, abs_score):

        return float(np.clip(abs_score / self.Z_VERY_STRONG, 0.0, 1.0))

    def _volatility_adjusted_confidence(self, confidence, volatility):

        if volatility <= 0:
            return confidence

        penalty = np.clip(volatility / 2.0, 0.0, 1.0)

        adjusted = confidence * (1.0 - 0.3 * penalty)

        return float(np.clip(adjusted, 0.0, 1.0))

    # ============================================================
    # POSITION SIZE
    # ============================================================

    def _suggest_position_size(self, confidence, risk_level, volatility_regime):

        base = confidence

        if risk_level == "elevated":
            base *= 0.70

        elif risk_level == "high":
            base *= 0.80

        if volatility_regime == "high_volatility":
            base *= 0.70

        elif volatility_regime == "low_volatility":
            base *= 1.10

        return float(np.clip(base, self.MIN_POSITION_SIZE, self.MAX_POSITION_SIZE))

    # ============================================================
    # MAIN ANALYSIS
    # ============================================================

    def analyze(self, context: Dict[str, Any]):

        if "row" in context:

            row = context.get("row", {}) or {}

            probability_stats = context.get("probability_stats")

            drift_state = context.get("drift_state")

            political_risk_label = context.get("political_risk_label")

        else:

            row = context or {}

            probability_stats = None

            drift_state = None

            political_risk_label = row.get("political_risk_label")

        raw_model_score = self._safe_float(
            row.get("raw_model_score",
                row.get("alpha_score",
                    row.get("score")))
        )

        final_score = float(
            np.clip(raw_model_score, -self.MAX_SCORE_SANITY, self.MAX_SCORE_SANITY)
        )

        abs_score = abs(final_score)

        alpha_strength = float(abs_score)

        signal = row.get("signal", "NEUTRAL")

        if signal not in {"LONG", "SHORT", "NEUTRAL"}:
            signal = "NEUTRAL"

        volatility = self._safe_float(row.get("volatility"), 0.0)

        rsi = np.clip(self._safe_float(row.get("rsi"), 50.0), 0, 100)

        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)

        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)

        regime_feat = self._safe_float(row.get("regime_feature"), 0.0)

        warnings: List[str] = []

        reasoning: List[str] = []

        # ---------------------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------------------

        confidence = self._confidence_numeric(abs_score)

        confidence = self._volatility_adjusted_confidence(confidence, volatility)

        # ---------------------------------------------------------
        # DISPERSION WARNING
        # ---------------------------------------------------------

        if probability_stats:

            std = self._safe_float(probability_stats.get("std"), 0.0)

            if std < 0.05:
                warnings.append("Low cross-sectional dispersion")

        # ---------------------------------------------------------
        # MOMENTUM CONTRADICTION (TEST REQUIRED)
        # ---------------------------------------------------------

        if signal == "LONG" and momentum_z < -0.5:
            warnings.append("Momentum contradicts LONG signal")

        if signal == "SHORT" and momentum_z > 0.5:
            warnings.append("Momentum contradicts SHORT signal")

        # ---------------------------------------------------------
        # TECHNICAL ALIGNMENT
        # ---------------------------------------------------------

        alignment = self._alignment_score(signal, momentum_z, ema_ratio)

        technical_score = alignment / 2.0

        # ---------------------------------------------------------
        # RSI EXTREMES
        # ---------------------------------------------------------

        if signal == "LONG" and rsi > self.RSI_OVERBOUGHT:
            warnings.append("RSI overbought — LONG entry risk elevated")

        if signal == "SHORT" and rsi < self.RSI_OVERSOLD:
            warnings.append("RSI oversold — SHORT entry risk elevated")

        # ---------------------------------------------------------
        # VOLATILITY REGIME
        # ---------------------------------------------------------

        if regime_feat > 1.5:

            volatility_regime = "high_volatility"

        elif regime_feat < -0.5:

            volatility_regime = "low_volatility"

        else:

            volatility_regime = "normal"

        # ---------------------------------------------------------
        # RISK LEVEL
        # ---------------------------------------------------------

        if volatility_regime == "high_volatility":

            risk_level = "elevated"

        elif abs_score < 0.5:

            risk_level = "high"

        elif abs_score < 1.0:

            risk_level = "moderate"

        else:

            risk_level = "low"

        # ---------------------------------------------------------
        # DRIFT PENALTY
        # ---------------------------------------------------------

        drift_flag = False

        if drift_state in {"soft", "hard"}:

            drift_flag = True

            confidence *= 0.75

            warnings.append(f"Drift detected: {drift_state}")

        # ---------------------------------------------------------
        # POLITICAL RISK
        # ---------------------------------------------------------

        if political_risk_label == "CRITICAL":

            signal = "NEUTRAL"

            warnings.append("Political risk CRITICAL — trading disabled")

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # ---------------------------------------------------------
        # AGENT SCORE
        # ---------------------------------------------------------

        agent_score = float(np.clip(

            0.5 * confidence
            + 0.3 * technical_score
            + 0.2 * (0.0 if drift_flag else 1.0),

            0.0, 1.0

        ))

        # ---------------------------------------------------------
        # TRADE APPROVAL
        # ---------------------------------------------------------

        conf_threshold = (

            self.LOW_VOL_CONFIDENCE
            if volatility_regime == "low_volatility"
            else self.MIN_CONFIDENCE_TO_TRADE

        )

        trade_approved = (

            signal != "NEUTRAL"
            and confidence > conf_threshold
            and not drift_flag
            and political_risk_label != "CRITICAL"

        )

        # ---------------------------------------------------------
        # POSITION SIZE
        # ---------------------------------------------------------

        position_size_hint = self._suggest_position_size(

            confidence,
            risk_level,
            volatility_regime

        )

        governance_score = int(np.clip(agent_score * 100, 0, 100))

        reasoning.append(
            f"Signal={signal} score={final_score:.2f} confidence={confidence:.2f}"
        )

        explanation = (
            f"{signal} | score={final_score:.2f} | conf={confidence:.2f} | "
            f"tech={technical_score:.2f} | risk={risk_level}"
        )

        hybrid_output = self._format_output(

            score=agent_score,

            confidence=confidence,

            signals={
                "direction": signal,
                "trade_approved": trade_approved,
            },

            warnings=warnings,

            reasoning=reasoning,

        )

        hybrid_output["component_scores"] = {

            "confidence": confidence,

            "technical_alignment": technical_score,

            "drift_penalty": 0.0 if drift_flag else 1.0,

        }

        return {

            "signal": signal,

            "score": agent_score,

            "alpha_strength": alpha_strength,

            "confidence_numeric": confidence,

            "technical_score": technical_score,

            "agent_score": agent_score,

            "risk_level": risk_level,

            "volatility_regime": volatility_regime,

            "alignment_score": alignment,

            "position_size_hint": position_size_hint,

            "trade_approved": trade_approved,

            "drift_flag": drift_flag,

            "governance_score": governance_score,

            "reasoning": reasoning,

            "warnings": warnings,

            "explanation": explanation,

            "hybrid": hybrid_output,

        }
