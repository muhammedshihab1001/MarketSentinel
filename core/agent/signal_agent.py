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
    # PUBLIC API
    ########################################################

    def analyze(
        self,
        row: Dict[str, Any],
        probability_stats: Dict[str, float],
    ) -> Dict[str, Any]:

        score = float(row["score"])
        rank_pct = float(row["rank_pct"])
        signal = row["signal"]

        volatility = float(row.get("volatility", 0.0))
        rsi = float(row.get("rsi", 50.0))
        ema_ratio = float(row.get("ema_ratio", 1.0))
        momentum_z = float(row.get("momentum_20_z", 0.0))
        regime_feature = float(row.get("regime_feature", 0.0))

        warnings: List[str] = []

        ####################################################
        # 0️⃣ Cross-sectional dispersion check (RESTORED)
        ####################################################

        std_dispersion = float(probability_stats.get("std", 0.0))
        if std_dispersion < self.DISPERSION_WEAK:
            warnings.append("Low cross-sectional dispersion detected.")

        ####################################################
        # 1️⃣ Confidence
        ####################################################

        if score >= self.HIGH_CONF_THRESHOLD:
            confidence = "high"
        elif score >= self.MOD_CONF_THRESHOLD:
            confidence = "moderate"
        elif score >= self.LOW_CONF_THRESHOLD:
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

        # Distance from neutral probability
        strength += abs(score - 0.5) * 100

        # Rank distance boost
        strength += abs(rank_pct - 0.5) * 50

        # Momentum alignment boost
        if (
            (signal == "LONG" and momentum_z > 0) or
            (signal == "SHORT" and momentum_z < 0)
        ):
            strength += 10

        # Trend alignment boost
        if (
            (signal == "LONG" and trend == "bullish") or
            (signal == "SHORT" and trend == "bearish")
        ):
            strength += 10

        # Volatility penalty
        if volatility_regime == "high_volatility":
            strength -= 10

        # Warning penalty
        strength -= len(warnings) * 5

        strength_score = int(np.clip(strength, 0, 100))

        ####################################################
        # 7️⃣ Risk Level (Adjusted Policy)
        ####################################################

        # High-confidence signals should map to low risk
        if confidence == "high" and strength_score >= 60:
            risk_level = "low"
        elif strength_score >= 75:
            risk_level = "low"
        elif strength_score >= 50:
            risk_level = "moderate"
        else:
            risk_level = "elevated"

        ####################################################
        # Explanation Text
        ####################################################

        explanation = (
            f"{signal} signal with {confidence} conviction "
            f"(score={score:.3f}, rank={rank_pct:.2f}). "
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