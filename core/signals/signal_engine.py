from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import os

from core.risk.position_sizer import PositionSizer


# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

@dataclass(frozen=True)
class SignalConfig:

    prob_threshold: float = float(
        os.getenv("PROB_THRESHOLD", "0.60")
    )

    sentiment_threshold: float = float(
        os.getenv("SENTIMENT_THRESHOLD", "0.10")
    )

    volatility_cap: float = float(
        os.getenv("VOL_CAP", "0.06")
    )

    volatility_throttle: float = float(
        os.getenv("VOL_THROTTLE", "0.04")
    )

    min_risk_adjusted_return: float = float(
        os.getenv("MIN_RAR", "0.40")
    )

    min_confidence: float = float(
        os.getenv("MIN_CONFIDENCE", "0.35")
    )

    portfolio_value: float = float(
        os.getenv("PORTFOLIO_VALUE", "100000")
    )

    max_position_pct: float = float(
        os.getenv("MAX_POSITION_PCT", "0.08")
    )

    global_kill_switch: bool = (
        os.getenv("GLOBAL_TRADING_DISABLED", "false")
        .lower() == "true"
    )


# ---------------------------------------------------
# UTIL
# ---------------------------------------------------

def _safe(v, fallback=0.0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return fallback
    return float(v)


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


# ---------------------------------------------------
# FORECAST INTERPRETER (UPGRADED)
# ---------------------------------------------------

class ForecastInterpreter:

    def interpret(
        self,
        predicted_return: float,
        sentiment: float,
        rsi: float,
        volatility: float,
        prob_up: float
    ) -> Tuple[str, float]:

        predicted_return = _safe(predicted_return)
        sentiment = _safe(sentiment)
        rsi = _safe(rsi, 50)
        volatility = max(_safe(volatility, 0.02), 1e-6)
        prob_up = _safe(prob_up, 0.5)

        # 🔥 Risk-adjusted return (Sharpe-lite)
        rar = predicted_return / volatility

        rsi_edge = abs(50 - rsi) / 50

        confidence = (
            prob_up * 0.55 +
            math.tanh(abs(rar)) * 0.25 +
            abs(sentiment) * 0.10 +
            rsi_edge * 0.10
        )

        confidence = _clamp(confidence, 0.0, 1.0)

        if rar > 0.4 and prob_up > 0.55:
            return "BUY", round(confidence, 3)

        if rar < -0.4 and prob_up < 0.45:
            return "SELL", round(confidence, 3)

        return "HOLD", round(confidence, 3)


# ---------------------------------------------------
# RISK GATE
# ---------------------------------------------------

class RiskGate:

    def __init__(self, config: SignalConfig):
        self.config = config

    def allow(
        self,
        signal: str,
        prob_up: float,
        volatility: float | None,
        regime: str | None
    ) -> bool:

        prob_up = _safe(prob_up, 0.5)
        volatility = _safe(volatility)
        regime = (regime or "").upper()

        if volatility > self.config.volatility_cap:
            return False

        if regime == "CRISIS":
            return False

        if signal == "BUY" and prob_up < self.config.prob_threshold:
            return False

        if signal == "SELL" and prob_up > (1 - self.config.prob_threshold):
            return False

        return True


# ---------------------------------------------------
# ENSEMBLE — NOW NON-DESTRUCTIVE
# ---------------------------------------------------

class EnsembleArbiter:

    def decide(
        self,
        base_signal: str,
        prob_up: float,
        lstm_prices: List[float],
        prophet_trend: str,
        config: SignalConfig
    ) -> str:

        # fallback to base signal if secondary models unavailable
        if not lstm_prices or len(lstm_prices) < 2:
            return base_signal

        first = max(_safe(lstm_prices[0], 1e-6), 1e-6)
        last = _safe(lstm_prices[-1], first)

        expected_return = (last - first) / first
        trend = str(prophet_trend).upper()

        if (
            base_signal == "BUY"
            and trend == "BULLISH"
            and expected_return > config.min_risk_adjusted_return * 0.02
        ):
            return "BUY"

        if (
            base_signal == "SELL"
            and trend == "BEARISH"
            and expected_return < -config.min_risk_adjusted_return * 0.02
        ):
            return "SELL"

        return base_signal


# ---------------------------------------------------
# MASTER ENGINE
# ---------------------------------------------------

class DecisionEngine:

    def __init__(self, config: SignalConfig | None = None):

        self.config = config or SignalConfig()

        self.interpreter = ForecastInterpreter()
        self.risk_gate = RiskGate(self.config)
        self.ensemble = EnsembleArbiter()
        self.position_sizer = PositionSizer()

    def generate(
        self,
        predicted_return: float,
        sentiment: float,
        rsi: float,
        prob_up: float,
        volatility: float,
        lstm_prices: List[float],
        prophet_trend: str,
        regime: str | None = None
    ) -> Dict:

        if self.config.global_kill_switch:
            return self._hold(0.0)

        base_signal, confidence = self.interpreter.interpret(
            predicted_return,
            sentiment,
            rsi,
            volatility,
            prob_up
        )

        if confidence < self.config.min_confidence:
            return self._hold(confidence)

        if not self.risk_gate.allow(
            base_signal,
            prob_up,
            volatility,
            regime
        ):
            return self._hold(confidence)

        final_signal = self.ensemble.decide(
            base_signal,
            prob_up,
            lstm_prices,
            prophet_trend,
            self.config
        )

        if final_signal == "HOLD":
            return self._hold(confidence)

        allocation = self.position_sizer.size_position(
            signal=final_signal,
            confidence=confidence,
            volatility=volatility,
            portfolio_value=self.config.portfolio_value
        )

        max_position = (
            self.config.portfolio_value *
            self.config.max_position_pct
        )

        allocation = min(allocation, max_position)

        position_pct = allocation / self.config.portfolio_value

        return {
            "signal": final_signal,
            "confidence": round(confidence, 3),
            "allocation": round(allocation, 2),
            "position_pct": round(position_pct, 4)
        }

    def _hold(self, confidence: float) -> Dict:

        return {
            "signal": "HOLD",
            "confidence": round(confidence, 3),
            "allocation": 0.0,
            "position_pct": 0.0
        }
