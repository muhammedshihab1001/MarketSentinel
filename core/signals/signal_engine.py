from dataclasses import dataclass
from typing import Dict, Optional
import math
import os
import time
import threading

from core.risk.position_sizer import PositionSizer
from core.portfolio.risk_engine import RiskEngine


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

    prob_threshold: float
    min_confidence: float
    crisis_volatility: float

    portfolio_value: float
    max_position_pct: float
    min_position_pct: float

    flip_cooldown_seconds: int
    global_kill_switch: bool

    enable_risk_engine: bool

    @staticmethod
    def load():

        cfg = SignalConfig(
            prob_threshold=_env_float("PROB_THRESHOLD", 0.55),
            min_confidence=_env_float("MIN_CONFIDENCE", 0.30),
            crisis_volatility=_env_float("CRISIS_VOL", 0.20),

            portfolio_value=_env_float("PORTFOLIO_VALUE", 100000),
            max_position_pct=_env_float("MAX_POSITION_PCT", 0.06),
            min_position_pct=_env_float("MIN_POSITION_PCT", 0.005),

            flip_cooldown_seconds=int(
                _env_float("FLIP_COOLDOWN_SECONDS", 60)
            ),

            global_kill_switch=_env_bool(
                "GLOBAL_TRADING_DISABLED",
                False
            ),

            enable_risk_engine=_env_bool(
                "ENABLE_RISK_ENGINE",
                False
            )
        )

        if not (0.5 <= cfg.prob_threshold <= 0.9):
            raise RuntimeError("Unsafe prob_threshold")

        if cfg.portfolio_value <= 0:
            raise RuntimeError("Invalid portfolio_value")

        if cfg.max_position_pct > 0.20:
            raise RuntimeError("Position cap too large.")

        return cfg


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
# DECISION ENGINE
###################################################

class DecisionEngine:

    def __init__(self, config: Optional[SignalConfig] = None):

        self.config = config or SignalConfig.load()

        self.position_sizer = PositionSizer()

        self._lock = threading.RLock()

        self._last_signal = {}
        self._last_flip_time = {}

    ###################################################

    def _flip_guard(self, ticker, new_signal):

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
            "confidence": round(confidence, 3),
            "allocation": 0.0,
            "position_pct": 0.0,
            "expected_value": 0.0,
            "risk_score": 0.0,
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
        price_df=None  # optional future hybrid support
    ) -> Dict:

        with self._lock:

            if self.config.global_kill_switch:
                return self._hold(0.0)

            prob_up = _safe(prob_up, 0.5)
            volatility = max(_safe(volatility, 0.02), 1e-6)

            ###################################################
            # CRISIS FILTER
            ###################################################

            if regime == "CRISIS" or volatility > self.config.crisis_volatility:
                return self._hold(0.2)

            ###################################################
            # PROBABILITY DECISION
            ###################################################

            if prob_up > self.config.prob_threshold:
                signal = "BUY"
            elif prob_up < (1 - self.config.prob_threshold):
                signal = "SELL"
            else:
                return self._hold(abs(prob_up - 0.5))

            confidence = _clamp(abs(prob_up - 0.5) * 2, 0.25, 0.95)

            if confidence < self.config.min_confidence:
                return self._hold(confidence)

            if not self._flip_guard(ticker, signal):
                return self._hold(confidence)

            ###################################################
            # OPTIONAL RISK ENGINE
            ###################################################

            risk_score = 0.0

            if self.config.enable_risk_engine and price_df is not None:

                risk = RiskEngine.analyze(price_df, signal)

                if risk["capital_multiplier"] == 0:
                    return self._hold(confidence)

                risk_score = risk["risk_score"]

            ###################################################
            # POSITION SIZING
            ###################################################

            allocation = self.position_sizer.size_position(
                signal=signal,
                confidence=confidence,
                volatility=volatility,
                portfolio_value=self.config.portfolio_value
            )

            if not math.isfinite(allocation) or allocation <= 0:
                return self._hold(confidence)

            max_alloc = self.config.portfolio_value * self.config.max_position_pct
            min_alloc = self.config.portfolio_value * self.config.min_position_pct

            allocation = _clamp(allocation, min_alloc, max_alloc)

            self._last_signal[ticker] = signal

            position_pct = allocation / self.config.portfolio_value

            return {
                "signal": signal,
                "confidence": round(confidence, 3),
                "allocation": round(allocation, 2),
                "position_pct": round(position_pct, 4),
                "expected_value": round(predicted_return, 5),
                "risk_score": round(risk_score, 4),
                "regime": regime
            }
