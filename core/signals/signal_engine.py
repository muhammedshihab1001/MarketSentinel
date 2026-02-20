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
        if not math.isfinite(v) or abs(v) > 1e6:
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
    base_min_edge: float
    smoothing_alpha: float

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
            base_min_edge=_env_float("MIN_PROB_EDGE", 0.08),
            smoothing_alpha=_env_float("PROB_SMOOTHING_ALPHA", 0.4)
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
# DECISION ENGINE (INSTITUTIONAL VERSION)
###################################################

class DecisionEngine:

    def __init__(self, config: Optional[SignalConfig] = None):

        self.config = config or SignalConfig.load()

        self._lock = threading.RLock()
        self._last_signal = {}
        self._last_flip_time = {}
        self._smoothed_prob = {}

    ###################################################

    def _regime_adjusted_edge(self, regime: Optional[str]) -> float:

        base = self.config.base_min_edge

        if regime == "BULL":
            return base * 0.8
        if regime == "BEAR":
            return base * 1.3
        if regime == "CRISIS":
            return 1.0

        return base

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

    def _smooth_probability(self, ticker, prob):

        prev = self._smoothed_prob.get(ticker, prob)

        alpha = self.config.smoothing_alpha
        smoothed = alpha * prob + (1 - alpha) * prev

        self._smoothed_prob[ticker] = smoothed

        return smoothed

    ###################################################

    def _hold(self, confidence: float) -> Dict:
        return {
            "signal": "HOLD",
            "confidence": round(float(confidence), 3),
            "edge": 0.0,
            "regime": None
        }

    ###################################################

    def generate(
        self,
        ticker: str,
        predicted_return: float,  # kept for compatibility (ignored)
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

            ###################################################
            # CRISIS FILTER
            ###################################################

            if regime == "CRISIS" or volatility > self.config.crisis_volatility:
                return self._hold(0.05)

            ###################################################
            # SMOOTH PROBABILITY
            ###################################################

            prob_up = self._smooth_probability(ticker, prob_up)

            ###################################################
            # EDGE CHECK (REGIME ADAPTIVE)
            ###################################################

            edge = abs(prob_up - 0.5)
            required_edge = self._regime_adjusted_edge(regime)

            if edge < required_edge:
                return self._hold(edge)

            ###################################################
            # LONG-ONLY SIGNAL
            ###################################################

            if prob_up <= 0.5:
                return self._hold(edge)

            ###################################################
            # CONFIDENCE MODEL
            ###################################################

            prob_strength = edge * 2.0
            vol_penalty = 1.0 / (1.0 + volatility * 4.0)

            confidence = prob_strength * vol_penalty
            confidence = _clamp(confidence, 0.05, 0.95)

            if not self._flip_guard(ticker, "BUY"):
                return self._hold(confidence)

            self._last_signal[ticker] = "BUY"

            return {
                "signal": "BUY",
                "confidence": round(confidence, 3),
                "edge": round(edge, 5),
                "regime": regime
            }