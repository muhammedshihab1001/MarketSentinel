import numpy as np
from typing import Dict, Any, List


class SignalAgent:

    HIGH_CONF_THRESHOLD = 0.75
    MOD_CONF_THRESHOLD = 0.60
    LOW_CONF_THRESHOLD = 0.52

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    EMA_BULLISH = 1.05
    EMA_BEARISH = 0.95

    MOMENTUM_STRONG = 1.0
    DISPERSION_WEAK = 0.02

    ########################################################
    # SAFE FLOAT
    ########################################################

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

    ########################################################
    # PUBLIC API
    ########################################################

    def analyze(
        self,
        row: Dict[str, Any],
        probability_stats: Dict[str, float],
    ) -> Dict[str, Any]:

        # 🔥 Regression alpha score (scaled 0–1)
        score = self._safe_float(row.get("score"), 0.5)
        rank_pct = self._safe_float(row.get("rank_pct"), 0.5)

        signal = row.get("signal", "NEUTRAL")

        volatility = self._safe_float(row.get("volatility"), 0.0)
        rsi = self._safe_float(row.get("rsi"), 50.0)
        ema_ratio = self._safe_float(row.get("ema_ratio"), 1.0)
        momentum_z = self._safe_float(row.get("momentum_20_z"), 0.0)
        regime_feature = self._safe_float(row.get("regime_feature"), 0.0)

        warnings: List[str] = []

        ####################################################
        # 0️⃣ Cross-sectional dispersion check
        ####################################################

        std_dispersion = self._safe_float(
            probability_stats.get("std"),
            0.0
        )

        if std_dispersion < self.DISPERSION_WEAK:
            warnings.append("Low cross-sectional signal dispersion detected.")

        ####################################################
        # 1️⃣ Confidence (based on cross-sectional rank)
        ####################################################

        if rank_pct >= 0.9:
            confidence = "very_high"
        elif rank_pct >= 0.75:
            confidence = "high"
        elif rank_pct >= 0.60:
            confidence = "moderate"
        elif rank_pct >= 0.50:
            confidence = "low"
        else:
            confidence = "very_low"

        ####################################################
        # 2️⃣ Volatility Context
        ####################################################

        if regime_feature == 1.0:
            volatility_regime = "high_volatility"
            warnings.append("High volatility regime detected.")
        else:
            volatility_regime = "normal"

        ####################################################
        # 3️⃣ RSI Context
        ####################################################

        if rsi > self.RSI_OVERBOUGHT:
            warnings.append("RSI indicates overbought condition.")
        elif rsi < self.RSI_OVERSOLD:
            warnings.append("RSI indicates oversold condition.")

        ####################################################
        # 4️⃣ Momentum Context
        ####################################################

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

        ####################################################
        # 5️⃣ Trend Context
        ####################################################

        if ema_ratio > self.EMA_BULLISH:
            trend = "bullish"
        elif ema_ratio < self.EMA_BEARISH:
            trend = "bearish"
        else:
            trend = "neutral"

        ####################################################
        # 6️⃣ Strength Score (0–100)
        ####################################################

        strength = 0.0

        # Alpha intensity (distance from neutral)
        strength += abs(score - 0.5) * 120

        # Rank importance
        strength += abs(rank_pct - 0.5) * 80

        # Momentum alignment boost
        if (
            (signal == "LONG" and momentum_z > 0) or
            (signal == "SHORT" and momentum_z < 0)
        ):
            strength += 15

        # Trend alignment boost
        if (
            (signal == "LONG" and trend == "bullish") or
            (signal == "SHORT" and trend == "bearish")
        ):
            strength += 10

        # Volatility penalty
        if volatility_regime == "high_volatility":
            strength -= 15

        # Warning penalty
        strength -= len(warnings) * 5

        strength_score = int(np.clip(strength, 0, 100))

        ####################################################
        # 7️⃣ Risk Level
        ####################################################

        if strength_score >= 80:
            risk_level = "low"
        elif strength_score >= 60:
            risk_level = "moderate"
        elif strength_score >= 40:
            risk_level = "elevated"
        else:
            risk_level = "high"

        ####################################################
        # Explanation Text (Production-ready)
        ####################################################

        explanation = (
            f"{signal} signal with {confidence} conviction "
            f"(alpha_score={score:.3f}, cross_rank={rank_pct:.2f}). "
            f"Trend: {trend}. "
            f"Volatility regime: {volatility_regime}. "
            f"Momentum: {momentum_state}."
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