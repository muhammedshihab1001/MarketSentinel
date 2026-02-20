import math
import os
import numpy as np
from dataclasses import dataclass


def _env_float(key: str, default: float) -> float:
    try:
        v = float(os.getenv(key, str(default)))
        if not math.isfinite(v):
            return default
        if abs(v) > 10:
            return default
        return v
    except Exception:
        return default


@dataclass(frozen=True)
class RiskConfig:

    max_position_size: float
    min_position_size: float
    volatility_target: float
    fractional_kelly: float
    max_gross_exposure_pct: float
    max_single_trade_dollars: float
    max_drawdown_kill: float

    @staticmethod
    def load():

        cfg = RiskConfig(
            max_position_size=_env_float("MAX_POSITION_SIZE", 0.06),
            min_position_size=_env_float("MIN_POSITION_SIZE", 0.005),
            volatility_target=_env_float("VOL_TARGET", 0.02),
            fractional_kelly=_env_float("FRACTIONAL_KELLY", 0.15),
            max_gross_exposure_pct=_env_float("MAX_GROSS_EXPOSURE", 0.70),
            max_single_trade_dollars=_env_float("MAX_SINGLE_TRADE", 25_000),
            max_drawdown_kill=_env_float("MAX_DRAWDOWN_KILL", 0.35),
        )

        if not (0 < cfg.max_position_size <= 0.20):
            raise RuntimeError("Unsafe max_position_size")

        if not (0 < cfg.max_gross_exposure_pct <= 1.5):
            raise RuntimeError("Unsafe gross exposure")

        if not (0 <= cfg.max_drawdown_kill <= 0.8):
            raise RuntimeError("Unsafe drawdown kill")

        return cfg


class PositionSizer:

    ABSOLUTE_MAX_POSITION = 0.15
    MIN_DEPLOYABLE_CAPITAL = 10_000
    VOL_FLOOR = 0.008
    VOL_CEILING = 0.12
    MAX_KELLY_FRACTION = 0.05

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig.load()

    @staticmethod
    def _safe(v, fallback):
        try:
            v = float(v)
            if not np.isfinite(v):
                return fallback
            return v
        except Exception:
            return fallback

    def _drawdown_scalar(self):
        raw = os.getenv("PORTFOLIO_DRAWDOWN_PCT", "0")
        try:
            dd = float(raw)
        except Exception:
            return 1.0

        if dd > 1:
            dd /= 100

        dd = min(max(dd, 0), 0.95)

        if dd < 0.05:
            return 1.0
        if dd < 0.10:
            return 0.80
        if dd < 0.20:
            return 0.50
        if dd < self.config.max_drawdown_kill:
            return 0.30

        return 0.0

    def _kelly_cap(self, edge, volatility, portfolio_value):

        if edge <= 0:
            return 0

        volatility = max(volatility, self.VOL_FLOOR)
        variance = volatility ** 2

        kelly_fraction = (edge / variance) * self.config.fractional_kelly
        kelly_fraction = min(kelly_fraction, self.MAX_KELLY_FRACTION)

        return portfolio_value * kelly_fraction

    def size_position(
        self,
        signal: str,
        confidence: float,
        volatility: float,
        portfolio_value: float,
        current_gross_exposure: float | None = None
    ) -> float:

        if portfolio_value < self.MIN_DEPLOYABLE_CAPITAL:
            return 0.0

        portfolio_value = self._safe(portfolio_value, 0)
        volatility = self._safe(volatility, self.config.volatility_target)

        volatility = max(
            self.VOL_FLOOR,
            min(volatility, self.VOL_CEILING)
        )

        dd_scalar = self._drawdown_scalar()
        if dd_scalar == 0:
            return 0.0

        direction = 1 if signal == "BUY" else -1

        # 🔐 FIXED EDGE CALCULATION
        edge = confidence

        vol_scalar = math.sqrt(
            self.config.volatility_target / volatility
        )

        raw_size = (
            portfolio_value *
            self.config.max_position_size *
            edge *
            vol_scalar *
            dd_scalar
        )

        kelly_cap = self._kelly_cap(
            edge,
            volatility,
            portfolio_value
        )

        if kelly_cap > 0:
            raw_size = min(raw_size, kelly_cap)

        config_cap = portfolio_value * self.config.max_position_size
        absolute_cap = portfolio_value * self.ABSOLUTE_MAX_POSITION
        trade_cap = self.config.max_single_trade_dollars

        final_size = min(
            raw_size,
            config_cap,
            absolute_cap,
            trade_cap
        )

        if current_gross_exposure is not None:
            max_allowed = (
                portfolio_value *
                self.config.max_gross_exposure_pct
            )
            remaining = max_allowed - current_gross_exposure
            final_size = min(final_size, max(remaining, 0))

        if final_size < portfolio_value * self.config.min_position_size:
            return 0.0

        return round(direction * float(final_size), 2)