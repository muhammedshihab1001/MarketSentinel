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
        os.getenv("PROB_THRESHOLD", "0.62")
    )

    sentiment_threshold: float = float(
        os.getenv("SENTIMENT_THRESHOLD", "0.12")
    )

    volatility_cap: float = float(
        os.getenv("VOL_CAP", "0.05")
    )

    volatility_throttle: float = float(
        os.getenv("VOL_THROTTLE", "0.035")
    )

    min_expected_return: float = float(
        os.getenv("MIN_EXPECTED_RETURN", "0.025")
    )

    min_confidence: float = float(
        os.getenv("MIN_CONFIDENCE", "0.30")
    )

    portfolio_value: float = float(
        os.getenv("PORTFOLIO_VALUE", "100000")
    )

    max_position_pct: float = float(
        os.getenv("MAX_POSITION_PCT", "0.08")
    )

    max_total_exposure_pct: float = float(
        os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.35")
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
# FORECAST INTERPRETER
# ---------------------------------------------------

class ForecastInterpreter:
    """
    Converts model output into normalized conviction.
    """

    def interpret(
        self,
        predicted_return: float,
        sentiment: float,
        rsi: float,
        volatility: float
    ) -> Tuple[str, float]:

        predicted_return = _safe(predicted_return)
        sentiment = _safe(sentiment)
        rsi = _safe(rsi, 50)
        volatility = _safe(volatility)

        rsi_component = math.tanh(abs(50 - rsi) / 25)

        base_confidence = (
            abs(predicted_return) * 0.45 +
            abs(sentiment) * 0.25 +
            rsi_component * 0.15
        )

        vol_penalty = min(volatility / 0.05, 1.0)
        confidence = base_confidence * (1 - 0.4 * vol_penalty)

        confidence = _clamp(confidence, 0.0, 1.0)

        if predicted_return > 0.025 and sentiment > 0 and rsi < 72:
            return "BUY", round(confidence, 3)

        if predicted_return < -0.025 and sentiment < 0:
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
# ENSEMBLE ARBITER
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

        prob_up = _safe(prob_up, 0.5)

        if not lstm_prices or len(lstm_prices) < 2:
            return "HOLD"

        first = max(_safe(lstm_prices[0], 1e-6), 1e-6)
        last = _safe(lstm_prices[-1], first)

        expected_return = (last - first) / first
        trend = str(prophet_trend).upper()

        if (
            base_signal == "BUY"
            and trend == "BULLISH"
            and expected_return > config.min_expected_return
            and prob_up >= config.prob_threshold
        ):
            return "BUY"

        if (
            base_signal == "SELL"
            and trend == "BEARISH"
            and expected_return < -config.min_expected_return
            and prob_up <= (1 - config.prob_threshold)
        ):
            return "SELL"

        return "HOLD"


# ---------------------------------------------------
# MASTER DECISION ENGINE
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
            volatility
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

        vol_scale = 1.0
        if volatility > self.config.volatility_throttle:
            vol_scale = max(
                0.25,
                1 - (volatility / 0.10)
            )

        allocation = self.position_sizer.size_position(
            signal=final_signal,
            confidence=confidence * vol_scale,
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


# ===================================================
# STRATEGY ENGINE
# ===================================================

class StrategyEngine:

    def top_opportunities(
        self,
        predictions: List[Dict],
        top_k: int = 5
    ):

        buys = [
            p for p in predictions
            if p.get("signal_today") == "BUY"
        ]

        buys.sort(
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )

        return buys[:top_k]

    def sell_alerts(
        self,
        predictions: List[Dict]
    ):

        return [
            p for p in predictions
            if p.get("signal_today") == "SELL"
            and p.get("confidence", 0) > 0.65
        ]

    def signal_distribution(
        self,
        predictions: List[Dict]
    ):

        dist = {"BUY": 0, "SELL": 0, "HOLD": 0}

        for p in predictions:
            signal = p.get("signal_today", "HOLD")
            dist[signal] += 1

        return dist
