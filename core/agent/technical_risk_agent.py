"""
MarketSentinel v4.1.1

Technical Risk Agent — evaluates market technical conditions and
risk factors to produce a gated quality score for the signal pipeline.

Portfolio-safe version for noisy market data (yfinance compatible).
"""

import numpy as np
from typing import Any, Dict, List

from core.agent.base_agent import BaseAgent


class TechnicalRiskAgent(BaseAgent):

    name = "TechnicalRiskAgent"

    weight = 0.7

    description = (
        "Evaluates momentum, EMA structure, RSI, volatility regime, "
        "and liquidity to produce a gated technical quality score."
    )

    # ─────────────────────────────────────────────
    # RSI THRESHOLDS
    # ─────────────────────────────────────────────

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    # ─────────────────────────────────────────────
    # VOLATILITY REGIME
    # ─────────────────────────────────────────────

    HIGH_VOL_THRESHOLD = 1.5
    LOW_VOL_THRESHOLD = -0.5

    # ─────────────────────────────────────────────
    # LIQUIDITY
    # ─────────────────────────────────────────────

    LOW_LIQUIDITY_THRESHOLD = 5e5

    # ─────────────────────────────────────────────
    # MOMENTUM
    # ─────────────────────────────────────────────

    MOMENTUM_STRONG = 1.0

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
    # MAIN ANALYZE
    # ============================================================

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:

        row = context.get("row", {})

        signal = row.get("signal", "NEUTRAL")

        drift_state = context.get("drift_state")

        warnings: List[str] = []

        reasoning: List[str] = []

        # ─────────────────────────────────────────────
        # FEATURE EXTRACTION
        # ─────────────────────────────────────────────

        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)

        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)

        rsi = np.clip(self._safe_float(row.get("rsi"), 50.0), 0, 100)

        regime_feat = self._safe_float(row.get("regime_feature"), 0.0)

        dollar_volume = max(
            self._safe_float(row.get("dollar_volume"), 0.0), 0.0
        )

        # ─────────────────────────────────────────────
        # MOMENTUM STRENGTH
        # ─────────────────────────────────────────────

        momentum_strength = float(np.clip(abs(momentum_z) / 3.0, 0.0, 1.0))

        if abs(momentum_z) < 0.5:
            warnings.append("Low momentum — directionless asset")

        if momentum_z > self.MOMENTUM_STRONG:
            bias = "bullish"
        elif momentum_z < -self.MOMENTUM_STRONG:
            bias = "bearish"
        else:
            bias = "neutral"

        reasoning.append(f"Momentum bias: {bias} (z={momentum_z:.2f})")

        # ─────────────────────────────────────────────
        # EMA STRUCTURE
        # ─────────────────────────────────────────────

        ema_score = float(np.clip(abs(ema_ratio - 1.0) * 5.0, 0.0, 1.0))

        if signal == "LONG" and ema_ratio < 1.0:
            warnings.append("EMA structure not supportive of LONG")

        if signal == "SHORT" and ema_ratio > 1.0:
            warnings.append("EMA structure not supportive of SHORT")

        reasoning.append(f"EMA ratio: {ema_ratio:.4f} | ema_score={ema_score:.2f}")

        # ─────────────────────────────────────────────
        # RSI SANITY
        # ─────────────────────────────────────────────

        if rsi > self.RSI_OVERBOUGHT:
            warnings.append(f"RSI overbought ({rsi:.1f})")
            rsi_score = 0.4
        elif rsi < self.RSI_OVERSOLD:
            warnings.append(f"RSI oversold ({rsi:.1f})")
            rsi_score = 0.4
        else:
            rsi_score = 1.0

        reasoning.append(f"RSI: {rsi:.1f} | rsi_score={rsi_score:.2f}")

        # ─────────────────────────────────────────────
        # VOLATILITY REGIME
        # ─────────────────────────────────────────────

        if regime_feat > self.HIGH_VOL_THRESHOLD:

            warnings.append("High volatility regime detected")

            vol_multiplier = 0.7

            volatility_regime = "high_volatility"

        elif regime_feat < self.LOW_VOL_THRESHOLD:

            reasoning.append("Low volatility stability regime")

            vol_multiplier = 1.1

            volatility_regime = "low_volatility"

        else:

            vol_multiplier = 1.0

            volatility_regime = "normal"

        # ─────────────────────────────────────────────
        # LIQUIDITY GUARD
        # ─────────────────────────────────────────────

        if dollar_volume < self.LOW_LIQUIDITY_THRESHOLD:

            warnings.append(
                f"Low liquidity — volume={dollar_volume:.0f}"
            )

            liquidity_score = 0.5

        else:

            liquidity_score = 1.0

        # ─────────────────────────────────────────────
        # DRIFT PENALTY
        # ─────────────────────────────────────────────

        drift_penalty = 1.0

        if drift_state == "soft":

            drift_penalty = 0.85

            warnings.append("Soft drift — signal quality reduced")

        elif drift_state == "hard":

            drift_penalty = 0.70

            warnings.append("Hard drift — signal quality heavily reduced")

        # ─────────────────────────────────────────────
        # SIGNAL SCORE
        # ─────────────────────────────────────────────

        signal_score = float(np.mean([

            momentum_strength,

            ema_score,

            rsi_score

        ]))

        # ─────────────────────────────────────────────
        # GATE MULTIPLIER
        # ─────────────────────────────────────────────

        gate_multiplier = liquidity_score * drift_penalty * vol_multiplier

        final_score = float(np.clip(signal_score * gate_multiplier, 0.0, 1.0))

        # ─────────────────────────────────────────────
        # CONFIDENCE
        # ─────────────────────────────────────────────

        indicator_votes = sum([

            1 if momentum_strength > 0.5 else 0,

            1 if ema_score > 0.3 else 0,

            1 if rsi_score == 1.0 else 0

        ])

        confidence = float(np.clip(

            (indicator_votes / 3.0) * gate_multiplier,

            0.0, 1.0

        ))

        governance_score = int(np.clip(final_score * 100, 0, 100))

        explanation = (
            f"tech_score={final_score:.2f} | conf={confidence:.2f} | "
            f"bias={bias} | regime={volatility_regime} | "
            f"drift={drift_state or 'none'}"
        )

        reasoning.append(
            f"signal_score={signal_score:.2f} | gate={gate_multiplier:.2f} | final={final_score:.2f}"
        )

        # ─────────────────────────────────────────────
        # FORMAT OUTPUT
        # ─────────────────────────────────────────────

        output = self._format_output(

            score=final_score,

            confidence=confidence,

            signals={

                "bias": bias,

                "volatility_regime": volatility_regime

            },

            warnings=warnings,

            reasoning=reasoning

        )

        output["bias"] = bias

        output["governance_score"] = governance_score

        output["explanation"] = explanation

        output["component_scores"] = {

            "momentum": momentum_strength,

            "ema": ema_score,

            "rsi": rsi_score,

            "signal_score": signal_score,

            "liquidity": liquidity_score,

            "drift_penalty": drift_penalty,

            "volatility_multiplier": vol_multiplier,

        }

        return output
