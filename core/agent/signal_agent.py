"""
MarketSentinel v4.1.0

Signal Agent — interprets XGBoost model output into a structured
trading signal with confidence, risk level, and position sizing hint.

Responsibilities:
    - Parse raw model score into directional signal
    - Score technical alignment (RSI, EMA, momentum)
    - Apply drift and political risk penalties
    - Produce hybrid consensus output via BaseAgent._format_output()
"""

import numpy as np
from typing import Any, Dict, List, Optional

from core.agent.base_agent import BaseAgent

EPSILON = 1e-12


class SignalAgent(BaseAgent):
    """
    Interprets ML model output + technical features into a
    structured trading signal compatible with hybrid consensus scoring.
    """

    name        = "SignalAgent"
    weight      = 1.0
    description = (
        "Interprets XGBoost model scores and technical indicators "
        "into directional signals with confidence and position sizing."
    )

    # ── Technical thresholds ─────────────────────────────────────────────────
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD   = 30

    EMA_BULLISH    = 1.05
    EMA_BEARISH    = 0.95

    # ── Score thresholds ──────────────────────────────────────────────────────
    Z_VERY_STRONG    = 2.0
    MAX_SCORE_SANITY = 10.0

    # ── Position sizing ───────────────────────────────────────────────────────
    MAX_POSITION_SIZE = 1.0
    MIN_POSITION_SIZE = 0.0

    # ── Trade approval gates ──────────────────────────────────────────────────
    MIN_CONFIDENCE_TO_TRADE = 0.30
    LOW_VOL_CONFIDENCE      = 0.20   # lower threshold in calm regimes

    # ─────────────────────────────────────────────────────────────────────────
    # TECHNICAL ALIGNMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _alignment_score(
        self,
        signal:     str,
        momentum_z: float,
        ema_ratio:  float,
    ) -> int:
        """
        Count how many technical indicators confirm the model signal.
        Returns 0, 1, or 2.
        """
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

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIDENCE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _confidence_numeric(self, abs_score: float) -> float:
        """Scale abs model score to [0, 1] confidence."""
        return float(np.clip(abs_score / self.Z_VERY_STRONG, 0.0, 1.0))

    def _volatility_adjusted_confidence(
        self,
        confidence: float,
        volatility: float,
    ) -> float:
        """Penalise confidence in high-volatility environments."""
        if volatility <= 0:
            return confidence
        penalty  = np.clip(volatility / 2.0, 0.0, 1.0)
        adjusted = confidence * (1.0 - 0.3 * penalty)
        return float(np.clip(adjusted, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # POSITION SIZE
    # ─────────────────────────────────────────────────────────────────────────

    def _suggest_position_size(
        self,
        confidence:        float,
        risk_level:        str,
        volatility_regime: str,
    ) -> float:
        """
        Scale confidence into a suggested position size in [0, 1].
        Higher risk / higher volatility → smaller position.
        """
        base = confidence

        # Risk-level scaling  (elevated > high > moderate > low risk)
        if risk_level == "elevated":
            base *= 0.70
        elif risk_level == "high":
            base *= 0.80
        # moderate / low → no penalty

        # Volatility-regime scaling
        if volatility_regime == "high_volatility":
            base *= 0.70
        elif volatility_regime == "low_volatility":
            base *= 1.10   # calm market → slightly larger sizing OK

        return float(np.clip(base, self.MIN_POSITION_SIZE, self.MAX_POSITION_SIZE))

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ANALYZE
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse model output + market features and produce a
        structured signal dict with hybrid consensus sub-output.

        Accepts context either as:
            {"row": {...}, "probability_stats": ..., ...}   (preferred)
            {flat feature dict}                              (legacy fallback)
        """

        # ── Context unpacking ─────────────────────────────────────────────────
        if "row" in context:
            row                  = context.get("row", {})
            probability_stats    = context.get("probability_stats")
            drift_score          = context.get("drift_score")
            drift_state          = context.get("drift_state")
            political_risk_label = context.get("political_risk_label")
        else:
            # Legacy flat-context fallback
            row                  = context
            probability_stats    = None
            drift_score          = None
            drift_state          = None
            political_risk_label = context.get("political_risk_label")

        # ── Raw model score ───────────────────────────────────────────────────
        raw_model_score = self._safe_float(
            row.get("raw_model_score",
                row.get("alpha_score",
                    row.get("score"))),
            default=0.0,
        )
        final_score = float(
            np.clip(raw_model_score, -self.MAX_SCORE_SANITY, self.MAX_SCORE_SANITY)
        )
        abs_score     = abs(final_score)
        alpha_strength = float(abs_score)

        # ── Signal direction ──────────────────────────────────────────────────
        signal = row.get("signal", "NEUTRAL")
        if signal not in {"LONG", "SHORT", "NEUTRAL"}:
            signal = "NEUTRAL"

        # ── Feature extraction (all safe — yfinance data can be noisy) ────────
        volatility  = self._safe_float(row.get("volatility"),       0.0)
        rsi         = self._safe_float(row.get("rsi"),              50.0)
        ema_ratio   = self._safe_float(row.get("ema_ratio"),        1.0)
        momentum_z  = self._safe_float(row.get("momentum_20_z"),    0.0)
        regime_feat = self._safe_float(row.get("regime_feature"),   0.0)

        warnings:  List[str] = []
        reasoning: List[str] = []

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = self._confidence_numeric(abs_score)
        confidence = self._volatility_adjusted_confidence(confidence, volatility)

        # ── Low dispersion warning ────────────────────────────────────────────
        if probability_stats:
            std = self._safe_float(probability_stats.get("std"), 0.0)
            if std < 0.05:
                warnings.append("Low cross-sectional dispersion")

        # ── Momentum contradiction ────────────────────────────────────────────
        if signal == "LONG"  and momentum_z < 0:
            warnings.append("Momentum contradicts LONG signal")
        if signal == "SHORT" and momentum_z > 0:
            warnings.append("Momentum contradicts SHORT signal")

        # ── Technical confirmation ────────────────────────────────────────────
        alignment       = self._alignment_score(signal, momentum_z, ema_ratio)
        technical_score = alignment / 2.0   # normalise to [0, 1]

        if signal == "LONG"  and rsi > self.RSI_OVERBOUGHT:
            warnings.append("RSI overbought — LONG entry risk elevated")
        if signal == "SHORT" and rsi < self.RSI_OVERSOLD:
            warnings.append("RSI oversold — SHORT entry risk elevated")

        # ── Volatility regime ─────────────────────────────────────────────────
        if regime_feat > 1.5:
            volatility_regime = "high_volatility"
        elif regime_feat < -0.5:
            volatility_regime = "low_volatility"
        else:
            volatility_regime = "normal"

        # ── Risk level ────────────────────────────────────────────────────────
        # elevated = high-volatility regime (regime-driven)
        # high / moderate / low = conviction-driven
        if volatility_regime == "high_volatility":
            risk_level = "elevated"
        elif abs_score < 0.5:
            risk_level = "high"
        elif abs_score < 1.0:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # ── Drift penalty ─────────────────────────────────────────────────────
        drift_flag = False
        if drift_state in {"soft", "hard"}:
            drift_flag  = True
            confidence *= 0.75
            warnings.append(f"Drift detected: state={drift_state}")
            reasoning.append(f"Confidence reduced 25% due to {drift_state} drift.")

        # ── Political risk override ───────────────────────────────────────────
        if political_risk_label == "CRITICAL":
            signal = "NEUTRAL"
            warnings.append("Political risk CRITICAL — signal overridden to NEUTRAL")
            reasoning.append("PoliticalRiskAgent override applied: trading disabled.")

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # ── Composite agent score ─────────────────────────────────────────────
        agent_score = float(np.clip(
            0.5 * confidence
            + 0.3 * technical_score
            + 0.2 * (0.0 if drift_flag else 1.0),
            0.0, 1.0,
        ))

        # ── Trade approval gate ───────────────────────────────────────────────
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

        # ── Position sizing ───────────────────────────────────────────────────
        position_size_hint = self._suggest_position_size(
            confidence, risk_level, volatility_regime
        )

        governance_score = int(np.clip(agent_score * 100, 0, 100))

        # ── Reasoning summary ─────────────────────────────────────────────────
        reasoning.append(
            f"Signal={signal} | model_score={final_score:.2f} | "
            f"confidence={confidence:.2f} | technical={technical_score:.2f} | "
            f"risk={risk_level} | regime={volatility_regime} | "
            f"drift={drift_state or 'none'} | "
            f"political={political_risk_label or 'none'}"
        )

        explanation = (
            f"{signal} | score={final_score:.2f} | conf={confidence:.2f} | "
            f"tech={technical_score:.2f} | risk={risk_level} | "
            f"drift={drift_state or 'none'} | "
            f"political={political_risk_label or 'none'} | "
            f"agent_score={agent_score:.2f}"
        )

        # ── Hybrid consensus sub-output (uses BaseAgent._format_output) ──────
        hybrid_output = self._format_output(
            score=agent_score,
            confidence=confidence,
            signals={
                "direction":     signal,
                "trade_approved": trade_approved,
            },
            warnings=warnings,
            reasoning=reasoning,
        )
        # Attach component breakdown for transparency
        hybrid_output["component_scores"] = {
            "confidence":          confidence,
            "technical_alignment": technical_score,
            "drift_penalty":       0.0 if drift_flag else 1.0,
        }

        # ── Full return dict ──────────────────────────────────────────────────
        return {
            "signal":             signal,
            "alpha_strength":     alpha_strength,
            "confidence_numeric": confidence,
            "technical_score":    technical_score,
            "agent_score":        agent_score,
            "risk_level":         risk_level,
            "volatility_regime":  volatility_regime,
            "alignment_score":    alignment,
            "position_size_hint": position_size_hint,
            "trade_approved":     trade_approved,
            "drift_flag":         drift_flag,
            "governance_score":   governance_score,
            "reasoning":          reasoning,      # insertion order preserved
            "warnings":           warnings,       # insertion order preserved
            "explanation":        explanation,
            "hybrid":             hybrid_output,
        }