import numpy as np
from typing import Dict, Any, List, Optional


class SignalAgent:

    # =========================================================
    # THRESHOLDS (Institutional Tunables)
    # =========================================================

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
    # CONFIDENCE NUMERIC
    # =========================================================

    def _confidence_numeric(self, abs_score: float) -> float:
        return float(np.clip(abs_score / self.Z_VERY_STRONG, 0.0, 1.0))

    # =========================================================
    # POSITION SIZE SUGGESTION
    # =========================================================

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

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        row: Dict[str, Any],
        probability_stats: Optional[Dict[str, float]] = None,
        drift_score: Optional[float] = None,
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
            final_score = np.sign(final_score) * self.MAX_SCORE_SANITY

        abs_score = abs(final_score)

        if abs_score >= self.Z_VERY_STRONG:
            confidence = "very_high"
        elif abs_score >= self.Z_STRONG:
            confidence = "high"
        elif abs_score >= self.Z_MODERATE:
            confidence = "moderate"
        else:
            confidence = "low"

        confidence_numeric = self._confidence_numeric(abs_score)

        reasoning.append(f"Alpha classified as {confidence} strength.")

        # =====================================================
        # 2️⃣ Technical Assessment
        # =====================================================

        if rsi > self.RSI_OVERBOUGHT:
            warnings.append("RSI overbought.")
        elif rsi < self.RSI_OVERSOLD:
            reasoning.append("Oversold condition supports reversal.")

        if ema_ratio > self.EMA_BULLISH:
            trend = "bullish"
        elif ema_ratio < self.EMA_BEARISH:
            trend = "bearish"
        else:
            trend = "neutral"

        if momentum_z > self.MOMENTUM_STRONG:
            momentum_state = "strong_positive"
        elif momentum_z < -self.MOMENTUM_STRONG:
            momentum_state = "strong_negative"
        else:
            momentum_state = "neutral"

        alignment = self._alignment_score(signal, momentum_z, ema_ratio)

        if alignment == 2:
            reasoning.append("Technical alignment strong.")
        elif alignment == 1:
            reasoning.append("Partial technical alignment.")
        else:
            warnings.append("Signal lacks technical confirmation.")

        # =====================================================
        # 3️⃣ Risk Assessment
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
        # 4️⃣ Macro Regime
        # =====================================================

        if breadth > 0.65:
            macro_regime = "risk_on"
        elif breadth < 0.35:
            macro_regime = "risk_off"
        else:
            macro_regime = "neutral"

        # =====================================================
        # 5️⃣ Cross-Sectional Health
        # =====================================================

        dispersion = 0.0
        percentile = None

        if probability_stats:
            dispersion = self._safe_float(probability_stats.get("std"), 0.0)
            percentile = probability_stats.get("percentile")

        if dispersion < self.DISPERSION_WEAK:
            warnings.append("Low cross-sectional dispersion.")

        # =====================================================
        # 6️⃣ Drift Awareness
        # =====================================================

        drift_flag = False

        if drift_score is not None:
            drift_score = self._safe_float(drift_score, 0.0)
            if drift_score > 0.5:
                drift_flag = True
                warnings.append("Feature distribution drift detected.")
                confidence_numeric *= 0.7

        # =====================================================
        # 7️⃣ Position Sizing Suggestion
        # =====================================================

        position_size_hint = self._suggest_position_size(
            confidence_numeric,
            risk_level,
            volatility_regime
        )

        trade_approved = (
            signal != "NEUTRAL"
            and confidence_numeric > 0.3
            and not drift_flag
        )

        # =====================================================
        # 8️⃣ Institutional Strength Score (0–100)
        # =====================================================

        strength_score = int(
            np.clip(abs_score * 40 + alignment * 5 + confidence_numeric * 20, 0, 100)
        )

        # =====================================================
        # 9️⃣ Explanation
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
        # RETURN STRUCTURED OUTPUT
        # =====================================================

        return {
            "signal": signal,
            "confidence": confidence,
            "confidence_numeric": confidence_numeric,
            "strength_score": strength_score,
            "risk_level": risk_level,
            "volatility_regime": volatility_regime,
            "trend": trend,
            "momentum_state": momentum_state,
            "macro_regime": macro_regime,
            "alignment_score": alignment,
            "position_size_hint": position_size_hint,
            "trade_approved": trade_approved,
            "drift_flag": drift_flag,
            "reasoning": reasoning,
            "warnings": warnings,
            "explanation": explanation,
        }