from dataclasses import dataclass
from typing import Dict, Optional
import math
import os
import time
import threading


###################################################
# SAFE ENV
###################################################

def _env_float(key: str, default: float) -> float:
    try:
        v = float(os.getenv(key, str(default)))
        if not math.isfinite(v):
            return default
        if abs(v) > 1e6:
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

    crisis_volatility: float
    flip_cooldown_seconds: int
    global_kill_switch: bool
    min_probability_edge: float

    @staticmethod
    def load():

        return SignalConfig(
            crisis_volatility=_env_float("CRISIS_VOL", 0.25),
            flip_cooldown_seconds=int(
                _env_float("FLIP_COOLDOWN_SECONDS", 0)
            ),
            global_kill_switch=_env_bool(
                "GLOBAL_TRADING_DISABLED",
                False
            ),
            min_probability_edge=_env_float("MIN_PROB_EDGE", 0.08)
        )


###################################################
# UTIL
###################################################

def _safe(v, fallback=0.0):
    try:
        v = float(v)
        if not math.isfinite(v):
            return fallback
        return v
    except Exception:
        return fallback


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


###################################################
# DECISION ENGINE (IMPROVED)
###################################################

class DecisionEngine:

    def __init__(self, config: Optional[SignalConfig] = None):

        self.config = config or SignalConfig.load()

        self._lock = threading.RLock()
        self._last_signal = {}
        self._last_flip_time = {}

    ###################################################

    def _flip_guard(self, ticker, new_signal):

        if self.config.flip_cooldown_seconds == 0:
            return True

        now = time.time()

        last_signal = self._last_signal.get(ticker, "HOLD")
        last_flip = self._last_flip_time.get(ticker, 0)

        if (
            new_signal != last_signal
            and now - last_flip
            < self.config.flip_cooldown_seconds
        ):
            return False

        if new_signal != last_signal:
            self._last_flip_time[ticker] = now

        return True

    ###################################################

    def _hold(self, confidence: float) -> Dict:
        return {
            "signal": "HOLD",
            "confidence": round(float(confidence), 3),
            "edge": 0.0,
            "expected_value": 0.0,
            "regime": None
        }

    ###################################################

    def generate(
        self,
        ticker: str,
        predicted_return: float,
        prob_up: float,
        volatility: float,
        regime: Optional[str] = None,
        price_df=None
    ) -> Dict:

        with self._lock:

            if self.config.global_kill_switch:
                return self._hold(0.0)

            prob_up = _clamp(_safe(prob_up, 0.5), 0.0, 1.0)
            volatility = max(_safe(volatility, 0.02), 1e-6)
            predicted_return = _safe(predicted_return, 0.0)

            ###################################################
            # CRISIS FILTER
            ###################################################

            if regime == "CRISIS" or volatility > self.config.crisis_volatility:
                return self._hold(0.1)

            ###################################################
            # PROBABILITY EDGE FILTER
            ###################################################

            edge = abs(prob_up - 0.5)

            if edge < self.config.min_probability_edge:
                return self._hold(edge)

            ###################################################
            # DIRECTION
            ###################################################

            signal = "BUY" if prob_up > 0.5 else "SELL"

            ###################################################
            # CONFIDENCE MODEL (IMPROVED)
            ###################################################

            # Base probability strength
            prob_strength = edge * 2.0  # 0 to 1

            # Penalize high volatility
            vol_penalty = 1.0 / (1.0 + volatility * 5.0)

            # Include predicted_return magnitude
            return_strength = min(abs(predicted_return) * 10.0, 1.0)

            confidence = prob_strength * 0.6 + return_strength * 0.4
            confidence *= vol_penalty

            confidence = _clamp(confidence, 0.05, 0.95)

            if not self._flip_guard(ticker, signal):
                return self._hold(confidence)

            self._last_signal[ticker] = signal

            return {
                "signal": signal,
                "confidence": round(confidence, 3),
                "edge": round(edge, 5),
                "expected_value": round(predicted_return, 6),
                "regime": regime
            }