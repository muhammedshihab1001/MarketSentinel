from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import os
import time

from core.risk.position_sizer import PositionSizer


###################################################
# SAFE ENV
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
# CONFIG
###################################################

@dataclass(frozen=True)
class SignalConfig:

    prob_threshold: float
    sentiment_threshold: float

    volatility_cap: float
    crisis_volatility: float
    volatility_throttle: float

    min_risk_adjusted_return: float
    min_confidence: float

    portfolio_value: float
    max_position_pct: float
    min_position_pct: float
    max_total_exposure_pct: float

    flip_cooldown_seconds: int

    global_kill_switch: bool

    ###################################################

    @staticmethod
    def load():

        return SignalConfig(
            prob_threshold=_env_float("PROB_THRESHOLD", 0.60),
            sentiment_threshold=_env_float("SENTIMENT_THRESHOLD", 0.08),

            volatility_cap=_env_float("VOL_CAP", 0.07),
            crisis_volatility=_env_float("CRISIS_VOL", 0.12),
            volatility_throttle=_env_float("VOL_THROTTLE", 0.045),

            min_risk_adjusted_return=_env_float("MIN_RAR", 0.35),
            min_confidence=_env_float("MIN_CONFIDENCE", 0.40),

            portfolio_value=_env_float("PORTFOLIO_VALUE", 100000),
            max_position_pct=_env_float("MAX_POSITION_PCT", 0.06),
            min_position_pct=_env_float("MIN_POSITION_PCT", 0.01),
            max_total_exposure_pct=_env_float("MAX_TOTAL_EXPOSURE_PCT", 0.35),

            flip_cooldown_seconds=int(
                _env_float("FLIP_COOLDOWN_SECONDS", 300)
            ),

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

    VOL_FLOOR = 1e-6
    RAR_CLAMP = 5.0

    def interpret(
        self,
        predicted_return: float,
        sentiment: float,
        rsi: float,
        volatility: float,
        prob_up: float,
        config: SignalConfig
    ) -> Tuple[str, float]:

        predicted_return = _safe(predicted_return)
        sentiment = _safe(sentiment, None)  # allow None
        rsi = _safe(rsi, 50.0)
        prob_up = _safe(prob_up, 0.5)

        volatility = max(_safe(volatility, 0.02), self.VOL_FLOOR)

        rar = predicted_return / volatility
        rar = _clamp(rar, -self.RAR_CLAMP, self.RAR_CLAMP)

        prob_threshold = config.prob_threshold

        if volatility > config.volatility_throttle:
            prob_threshold += 0.03

        if volatility > config.crisis_volatility:
            prob_threshold += 0.05

        ################################################
        # DYNAMIC CONFIDENCE WEIGHTING
        ################################################

        rsi_edge = abs(50 - rsi) / 50

        weights = {
            "prob": 0.50,
            "rar": 0.35,
            "rsi": 0.15,
            "sentiment": 0.0
        }

        # Only activate sentiment if meaningful
        if sentiment is not None and abs(sentiment) > config.sentiment_threshold:
            weights["sentiment"] = 0.10
            weights["prob"] -= 0.05
            weights["rar"] -= 0.05

        confidence = (
            prob_up * weights["prob"] +
            math.tanh(rar) * weights["rar"] +
            rsi_edge * weights["rsi"] +
            (abs(sentiment) if sentiment else 0.0) * weights["sentiment"]
        )

        confidence = _clamp(confidence, 0.0, 1.0)

        ################################################
        # BUY / SELL LOGIC
        ################################################

        if rar > config.min_risk_adjusted_return and prob_up >= prob_threshold:
            return "BUY", round(confidence, 3)

        if rar < -config.min_risk_adjusted_return and prob_up <= (1 - prob_threshold):
            return "SELL", round(confidence, 3)

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

        prob_up = _safe(prob_up, 0.5)
        volatility = max(_safe(volatility, 0.02), 1e-6)
        regime = (regime or "").upper()

        if volatility > self.config.crisis_volatility:
            return False

        if regime == "CRISIS":
            return False

        if signal == "BUY" and prob_up < self.config.prob_threshold:
            return False

        return True


###################################################
# DECISION ENGINE
###################################################

class DecisionEngine:

    def __init__(self, config: SignalConfig | None = None):

        self.config = config or SignalConfig.load()

        self.interpreter = ForecastInterpreter()
        self.risk_gate = RiskGate(self.config)
        self.position_sizer = PositionSizer()

        self._last_signal = "HOLD"
        self._last_flip_time = 0

    ###################################################

    def _flip_guard(self, new_signal):

        now = time.time()

        if (
            new_signal != self._last_signal
            and now - self._last_flip_time
            < self.config.flip_cooldown_seconds
        ):
            return False

        if new_signal != self._last_signal:
            self._last_flip_time = now

        return True

    ###################################################

    def _hold(self, confidence: float) -> Dict:

        return {
            "signal": "HOLD",
            "confidence": round(confidence, 3),
            "allocation": 0.0,
            "position_pct": 0.0
        }

    ###################################################

    def generate(
        self,
        predicted_return: float,
        sentiment: float | None,
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
            prob_up,
            self.config
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

        if not self._flip_guard(base_signal):
            return self._hold(confidence)

        allocation = self.position_sizer.size_position(
            signal=base_signal,
            confidence=confidence,
            volatility=max(volatility, 1e-6),
            portfolio_value=self.config.portfolio_value
        )

        if not math.isfinite(allocation) or allocation <= 0:
            return self._hold(confidence)

        self._last_signal = base_signal

        position_pct = allocation / self.config.portfolio_value

        return {
            "signal": base_signal,
            "confidence": round(confidence, 3),
            "allocation": round(allocation, 2),
            "position_pct": round(position_pct, 4)
        }
