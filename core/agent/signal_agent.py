import numpy as np
from typing import Dict, Any, List


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

    # =========================================================
    # SAFE FLOAT
    # =========================================================

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

    # =========================================================
    # ALIGNMENT CHECK
    # =========================================================

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

        return alignment  # 0–2

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        row: Dict[str, Any],
        probability_stats: Dict[str, float] | None,
    ) -> Dict[str, Any]:

        # -----------------------------------------------------
        # Core Inputs
        # -----------------------------------------------------

        raw_model_score = self._safe_float(
            row.get("raw_model_score", row.get("alpha_score", row.get("score"))),
            0.0
        )

        final_score = self._safe_float(row.get("score"), 0.0)
        signal = row.get("signal", "NEUTRAL")

        volatility = self._safe_float(row.get("volatility"), 0.0)
        rsi = self._safe_float(row.get("rsi"), 50.0)
        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)
        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)
        regime_feature = self._safe_float(row.get("regime_feature"), 0.0)
        breadth = self._safe_float(row.get("breadth"), 0.5)

        warnings: List[str] = []
        reasoning: List[str] = []

        # =====================================================
        # 1️⃣ Score Sanity
        # =====================================================

        if abs(final_score) > self.MAX_SCORE_SANITY:
            warnings.append("Score unusually large.")
            reasoning.append("Score clipped for sanity monitoring.")

        abs_score = abs(final_score)

        if abs_score >= self.Z_VERY_STRONG:
            confidence = "very_high"
        elif abs_score >= self.Z_STRONG:
            confidence = "high"
        elif abs_score >= self.Z_MODERATE:
            confidence = "moderate"
        else:
            confidence = "low"

        reasoning.append(f"Alpha classified as {confidence} strength.")

        # =====================================================
        # 2️⃣ Technical Assessment
        # =====================================================

        # RSI
        if rsi > self.RSI_OVERBOUGHT:
            warnings.append("RSI overbought.")
            reasoning.append("Overbought condition detected.")
        elif rsi < self.RSI_OVERSOLD:
            reasoning.append("Oversold condition supports reversal potential.")

        # Trend
        if ema_ratio > self.EMA_BULLISH:
            trend = "bullish"
        elif ema_ratio < self.EMA_BEARISH:
            trend = "bearish"
        else:
            trend = "neutral"

        reasoning.append(f"Trend regime: {trend}.")

        # Momentum
        if momentum_z > self.MOMENTUM_STRONG:
            momentum_state = "strong_positive"
        elif momentum_z < -self.MOMENTUM_STRONG:
            momentum_state = "strong_negative"
        else:
            momentum_state = "neutral"

        reasoning.append(f"Momentum state: {momentum_state}.")

        if signal == "LONG" and momentum_z < 0:
            warnings.append("Momentum contradicts LONG signal.")
        if signal == "SHORT" and momentum_z > 0:
            warnings.append("Momentum contradicts SHORT signal.")

        # =====================================================
        # 3️⃣ Risk Assessment
        # =====================================================

        if regime_feature > 1.5:
            volatility_regime = "high_volatility"
            warnings.append("High volatility regime detected.")
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

        reasoning.append(f"Risk level assessed as {risk_level}.")

        # =====================================================
        # 4️⃣ Macro Assessment
        # =====================================================

        if breadth > 0.65:
            macro_regime = "risk_on"
        elif breadth < 0.35:
            macro_regime = "risk_off"
        else:
            macro_regime = "neutral"

        reasoning.append(f"Macro regime: {macro_regime}.")

        # =====================================================
        # 5️⃣ Cross-Sectional Health
        # =====================================================

        dispersion = 0.0
        if probability_stats:
            dispersion = self._safe_float(probability_stats.get("std"), 0.0)

        if dispersion < self.DISPERSION_WEAK:
            warnings.append("Low cross-sectional dispersion.")
            reasoning.append("Signal environment weak due to low dispersion.")

        # =====================================================
        # 6️⃣ Alignment Score
        # =====================================================

        alignment = self._alignment_score(signal, momentum_z, ema_ratio)

        if alignment == 2:
            reasoning.append("Technical alignment strong.")
        elif alignment == 1:
            reasoning.append("Partial technical alignment.")
        else:
            warnings.append("Signal lacks technical confirmation.")

        # =====================================================
        # 7️⃣ Strength Score (0–100 institutional scale)
        # =====================================================

        strength_score = int(np.clip(abs_score * 40 + alignment * 5, 0, 100))

        # =====================================================
        # 8️⃣ Explanation Summary
        # =====================================================

        explanation = (
            f"{signal} | raw={raw_model_score:.2f} | "
            f"final={final_score:.2f} | "
            f"confidence={confidence} | "
            f"trend={trend} | "
            f"macro={macro_regime} | "
            f"risk={risk_level}"
        )

        # =====================================================
        # STRUCTURED OUTPUT
        # =====================================================

        return {
            "signal": signal,
            "confidence": confidence,
            "strength_score": strength_score,
            "risk_level": risk_level,
            "volatility_regime": volatility_regime,
            "trend": trend,
            "momentum_state": momentum_state,
            "macro_regime": macro_regime,
            "alignment_score": alignment,
            "reasoning": reasoning,
            "warnings": warnings,
            "explanation": explanation,
        }