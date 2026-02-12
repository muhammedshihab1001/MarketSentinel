from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import os

from core.risk.position_sizer import PositionSizer


###################################################
# SAFE ENV PARSERS
###################################################

def _env_float(key: str, default: float) -> float:
    try:
        v = float(os.getenv(key, str(default)))
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes")


###################################################
# CONFIGURATION
###################################################

@dataclass(frozen=True)
class SignalConfig:

    prob_threshold: float
    sentiment_threshold: float

    volatility_cap: float
    volatility_throttle: float

    min_risk_adjusted_return: float
    min_confidence: float

    portfolio_value: float
    max_position_pct: float

    global_kill_switch: bool

    ###################################################

    @staticmethod
    def load():

        return SignalConfig(
            prob_threshold=_env_float("PROB_THRESHOLD", 0.60),
            sentiment_threshold=_env_float("SENTIMENT_THRESHOLD", 0.10),
            volatility_cap=_env_float("VOL_CAP", 0.06),
            volatility_throttle=_env_float("VOL_THROTTLE", 0.04),
            min_risk_adjusted_return=_env_float("MIN_RAR", 0.40),
            min_confidence=_env_float("MIN_CONFIDENCE", 0.35),
            portfolio_value=_env_float("PORTFOLIO_VALUE", 100000),
            max_position_pct=_env_float("MAX_POSITION_PCT", 0.08),
            global_kill_switch=_env_bool(
                "GLOBAL_TRADING_DISABLED",
                False
            )
        )


###################################################
# UTIL
###################################################

def _safe(v, fallback=0.0):

    if v is None:
        return fallback

    try:
        v = float(v)
    except Exception:
        return fallback

    if not math.isfinite(v):
        return fallback

    return v


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


###################################################
# FORECAST INTERPRETER
###################################################

class ForecastInterpreter:

    VALID_SIGNALS = {"BUY", "HOLD"}

    BUY_THRESHOLD_HIGH = 0.62
    BUY_THRESHOLD_LOW = 0.57

    RAR_STRONG = 0.65
    RAR_WEAK = 0.35

    VOL_FLOOR = 1e-6

    ###################################################

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
        rsi = _safe(rsi, 50.0)
        prob_up = _safe(prob_up, 0.5)

        volatility = max(_safe(volatility, 0.02), self.VOL_FLOOR)

        rar = predicted_return / volatility
        rsi_edge = abs(50.0 - rsi) / 50.0

        confidence = (
            prob_up * 0.50 +
            math.tanh(abs(rar)) * 0.30 +
            abs(sentiment) * 0.10 +
            rsi_edge * 0.10
        )

        confidence = _clamp(confidence, 0.0, 1.0)

        if prob_up >= self.BUY_THRESHOLD_HIGH and rar > self.RAR_WEAK:
            return "BUY", round(confidence, 3)

        if prob_up >= self.BUY_THRESHOLD_LOW and rar > self.RAR_STRONG:
            return "BUY", round(confidence, 3)

        return "HOLD", round(confidence, 3)


###################################################
# RISK GATE
###################################################

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

        if signal not in ForecastInterpreter.VALID_SIGNALS:
            raise RuntimeError(f"Invalid signal emitted: {signal}")

        prob_up = _safe(prob_up, 0.5)
        volatility = max(_safe(volatility, 0.02), 1e-6)
        regime = (regime or "").upper()

        if volatility > self.config.volatility_cap:
            return False

        if regime == "CRISIS":
            return False

        if signal == "BUY" and prob_up < self.config.prob_threshold:
            return False

        return True


###################################################
# ENSEMBLE CONFIRMATION
###################################################

class EnsembleArbiter:

    def decide(
        self,
        base_signal: str,
        prob_up: float,
        lstm_prices: List[float] | None,
        macro_trend: str | None,
        config: SignalConfig
    ) -> str:

        if base_signal == "HOLD":
            return "HOLD"

        if not lstm_prices or len(lstm_prices) < 2:
            return base_signal

        first = max(_safe(lstm_prices[0], 1e-6), 1e-6)
        last = _safe(lstm_prices[-1], first)

        expected_return = (last - first) / first
        macro_trend = (macro_trend or "").upper()

        if macro_trend not in ("BULLISH", "SIDEWAYS"):
            return "HOLD"

        if expected_return <= 0:
            return "HOLD"

        if prob_up < config.prob_threshold:
            return "HOLD"

        return "BUY"


###################################################
# MASTER ENGINE
###################################################

class DecisionEngine:

    def __init__(self, config: SignalConfig | None = None):

        self.config = config or SignalConfig.load()

        self.interpreter = ForecastInterpreter()
        self.risk_gate = RiskGate(self.config)
        self.ensemble = EnsembleArbiter()
        self.position_sizer = PositionSizer()

        self._last_signal = "HOLD"

    ###################################################

    def generate(
        self,
        predicted_return: float,
        sentiment: float,
        rsi: float,
        prob_up: float,
        volatility: float,
        lstm_prices: List[float] | None,
        macro_trend: str | None,
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
            macro_trend,
            self.config
        )

        # prevent flip-flop instability
        if self._last_signal == "BUY" and final_signal == "HOLD":
            return self._hold(confidence * 0.8)

        if final_signal == "HOLD":
            self._last_signal = "HOLD"
            return self._hold(confidence)

        allocation = self.position_sizer.size_position(
            signal=final_signal,
            confidence=confidence,
            volatility=max(volatility, 1e-6),
            portfolio_value=self.config.portfolio_value
        )

        max_position = (
            self.config.portfolio_value *
            self.config.max_position_pct
        )

        allocation = min(allocation, max_position)

        position_pct = allocation / self.config.portfolio_value

        self._last_signal = final_signal

        return {
            "signal": final_signal,
            "confidence": round(confidence, 3),
            "allocation": round(allocation, 2),
            "position_pct": round(position_pct, 4)
        }

    ###################################################

    def _hold(self, confidence: float) -> Dict:

        return {
            "signal": "HOLD",
            "confidence": round(confidence, 3),
            "allocation": 0.0,
            "position_pct": 0.0
        }
