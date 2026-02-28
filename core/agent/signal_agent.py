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
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        row: Dict[str, Any],
        probability_stats: Dict[str, float],
    ) -> Dict[str, Any]:

        score = self._safe_float(row.get("score"), 0.0)
        signal = row.get("signal", "NEUTRAL")

        volatility = self._safe_float(row.get("volatility"), 0.0)
        rsi = self._safe_float(row.get("rsi"), 50.0)
        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)
        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)
        regime_feature = self._safe_float(row.get("regime_feature"), 0.0)

        warnings: List[str] = []

        # =====================================================
        # 1️⃣ Confidence based on Z-score magnitude
        # =====================================================

        abs_score = abs(score)

        if abs_score >= self.Z_VERY_STRONG:
            confidence = "very_high"
        elif abs_score >= self.Z_STRONG:
            confidence = "high"
        elif abs_score >= self.Z_MODERATE:
            confidence = "moderate"
        else:
            confidence = "low"

        # =====================================================
        # 2️⃣ Volatility Regime
        # =====================================================

        if regime_feature == 1.0:
            volatility_regime = "high_volatility"
            warnings.append("High volatility regime detected.")
        else:
            volatility_regime = "normal"

        # =====================================================
        # 3️⃣ RSI Context
        # =====================================================

        if rsi > self.RSI_OVERBOUGHT:
            warnings.append("RSI overbought.")
        elif rsi < self.RSI_OVERSOLD:
            warnings.append("RSI oversold.")

        # =====================================================
        # 4️⃣ Momentum Context
        # =====================================================

        if momentum_z > self.MOMENTUM_STRONG:
            momentum_state = "strong_positive"
        elif momentum_z < -self.MOMENTUM_STRONG:
            momentum_state = "strong_negative"
        else:
            momentum_state = "neutral"

        if signal == "LONG" and momentum_z < 0:
            warnings.append("Momentum contradicts LONG signal.")

        if signal == "SHORT" and momentum_z > 0:
            warnings.append("Momentum contradicts SHORT signal.")

        # =====================================================
        # 5️⃣ Trend Context
        # =====================================================

        if ema_ratio > self.EMA_BULLISH:
            trend = "bullish"
        elif ema_ratio < self.EMA_BEARISH:
            trend = "bearish"
        else:
            trend = "neutral"

        # =====================================================
        # 6️⃣ Cross-sectional dispersion health
        # =====================================================

        dispersion = self._safe_float(
            probability_stats.get("std"),
            0.0
        )

        if dispersion < self.DISPERSION_WEAK:
            warnings.append("Low cross-sectional dispersion.")

        # =====================================================
        # 7️⃣ Strength Score (based purely on Z-score magnitude)
        # =====================================================

        strength_score = int(
            np.clip(abs_score * 40, 0, 100)
        )

        # =====================================================
        # 8️⃣ Risk Level
        # =====================================================

        if volatility_regime == "high_volatility":
            risk_level = "elevated"
        elif abs_score < 0.5:
            risk_level = "high"
        elif abs_score < 1.0:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # =====================================================
        # Explanation
        # =====================================================

        explanation = (
            f"{signal} signal | alpha_z={score:.2f} | "
            f"confidence={confidence}. "
            f"Trend={trend}, Momentum={momentum_state}, "
            f"VolatilityRegime={volatility_regime}."
        )

        return {
            "confidence": confidence,
            "volatility_regime": volatility_regime,
            "trend": trend,
            "momentum_state": momentum_state,
            "strength_score": strength_score,
            "risk_level": risk_level,
            "warnings": warnings,
            "explanation": explanation,
        }