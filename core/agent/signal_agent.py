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

    def __init__(self):
        pass

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

        if abs(rank_pct - 0.5) < 0.05:
            warnings.append("Signal near decision boundary.")

        if probability_stats["std"] < self.DISPERSION_WEAK:
            warnings.append("Low cross-sectional dispersion detected.")

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
        # Explanation Text (Deterministic)
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
            "warnings": warnings,
            "explanation": explanation,
        }