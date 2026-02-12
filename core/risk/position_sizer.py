from dataclasses import dataclass
import math
import os
import numpy as np


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


###################################################
# CONFIG
###################################################

@dataclass(frozen=True)
class RiskConfig:

    max_position_size: float
    min_position_size: float

    volatility_target: float
    confidence_boost: float

    max_gross_exposure_pct: float
    max_single_trade_dollars: float

    # ⭐ NEW
    fractional_kelly: float
    volatility_shock_level: float
    correlation_heat_cap: float

    ###################################################

    @staticmethod
    def load():

        return RiskConfig(
            max_position_size=_env_float("MAX_POSITION_SIZE", 0.08),
            min_position_size=_env_float("MIN_POSITION_SIZE", 0.01),
            volatility_target=_env_float("VOL_TARGET", 0.02),
            confidence_boost=_env_float("CONFIDENCE_BOOST", 1.35),
            max_gross_exposure_pct=_env_float("MAX_GROSS_EXPOSURE", 0.40),
            max_single_trade_dollars=_env_float("MAX_SINGLE_TRADE", 25_000),

            # ⭐ Institutional additions
            fractional_kelly=_env_float("FRACTIONAL_KELLY", 0.25),
            volatility_shock_level=_env_float("VOL_SHOCK_LEVEL", 0.08),
            correlation_heat_cap=_env_float("HEAT_CAP", 0.22),
        )


###################################################
# POSITION SIZER
###################################################

class PositionSizer:

    ABSOLUTE_MAX_POSITION = 0.15
    MIN_DEPLOYABLE_CAPITAL = 10_000

    VOL_FLOOR = 0.006
    VOL_CEILING = 0.12

    MAX_VOL_SCALAR = 1.4
    MIN_CONFIDENCE_FLOOR = 0.20

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig.load()

    ###################################################

    @staticmethod
    def _safe(v, fallback):

        try:
            v = float(v)
        except Exception:
            return fallback

        if not np.isfinite(v):
            return fallback

        return v

    ###################################################
    # DRAW DOWN SCALAR
    ###################################################

    def _drawdown_scalar(self):

        raw = os.getenv("PORTFOLIO_DRAWDOWN_PCT", "0")

        try:
            drawdown = float(raw)
        except Exception:
            return 1.0

        if drawdown > 1:
            drawdown /= 100.0

        drawdown = abs(drawdown)
        drawdown = min(drawdown, 0.80)

        if drawdown < 0.05:
            return 1.0
        if drawdown < 0.10:
            return 0.70
        if drawdown < 0.20:
            return 0.45

        return 0.20

    ###################################################
    # VOL SHOCK SCALAR (⭐ VERY IMPORTANT)
    ###################################################

    def _volatility_shock_scalar(self, volatility):

        if volatility <= self.config.volatility_shock_level:
            return 1.0

        # nonlinear cut
        ratio = self.config.volatility_shock_level / volatility

        return max(0.25, ratio)

    ###################################################
    # FRACTIONAL KELLY LIMITER
    ###################################################

    def _kelly_cap(self, confidence, volatility, portfolio_value):

        edge = confidence - 0.5
        edge = max(edge, 0)

        if edge <= 0:
            return 0

        kelly_fraction = (edge / max(volatility, 1e-6))

        kelly_fraction *= self.config.fractional_kelly

        kelly_fraction = min(kelly_fraction, 0.10)

        return portfolio_value * kelly_fraction

    ###################################################
    # POSITION SIZING
    ###################################################

    def size_position(
        self,
        signal: str,
        confidence: float,
        volatility: float | None,
        portfolio_value: float,
        current_gross_exposure: float | None = None,
        current_heat: float | None = None   # ⭐ NEW
    ) -> float:

        if signal != "BUY":
            return 0.0

        portfolio_value = self._safe(portfolio_value, 0)

        if portfolio_value < self.MIN_DEPLOYABLE_CAPITAL:
            return 0.0

        confidence = max(
            self._safe(confidence, 0.5),
            self.MIN_CONFIDENCE_FLOOR
        )

        volatility = self._safe(
            volatility,
            self.config.volatility_target
        )

        volatility = max(
            self.VOL_FLOOR,
            min(volatility, self.VOL_CEILING)
        )

        ###################################################
        # HEAT LIMIT
        ###################################################

        if current_heat is not None:

            if current_heat >= self.config.correlation_heat_cap:
                return 0.0

        ###################################################
        # EXPOSURE LIMIT
        ###################################################

        if current_gross_exposure is not None:

            max_allowed = (
                portfolio_value *
                self.config.max_gross_exposure_pct
            )

            if current_gross_exposure >= max_allowed:
                return 0.0

        ###################################################
        # SCALARS
        ###################################################

        confidence_scalar = math.tanh(
            confidence * self.config.confidence_boost
        )

        vol_scalar = math.sqrt(
            self.config.volatility_target / volatility
        )

        vol_scalar = min(vol_scalar, self.MAX_VOL_SCALAR)

        dd_scalar = self._drawdown_scalar()
        shock_scalar = self._volatility_shock_scalar(volatility)

        ###################################################
        # RAW SIZE
        ###################################################

        raw_size = (
            portfolio_value *
            self.config.max_position_size *
            confidence_scalar *
            vol_scalar *
            dd_scalar *
            shock_scalar
        )

        ###################################################
        # KELLY CAP
        ###################################################

        kelly_cap = self._kelly_cap(
            confidence,
            volatility,
            portfolio_value
        )

        if kelly_cap > 0:
            raw_size = min(raw_size, kelly_cap)

        ###################################################
        # CAPS
        ###################################################

        config_cap = portfolio_value * self.config.max_position_size
        absolute_cap = portfolio_value * self.ABSOLUTE_MAX_POSITION
        trade_cap = self.config.max_single_trade_dollars

        capped_size = min(
            raw_size,
            config_cap,
            absolute_cap,
            trade_cap
        )

        ###################################################
        # GROSS FINAL
        ###################################################

        if current_gross_exposure is not None:

            remaining = (
                portfolio_value *
                self.config.max_gross_exposure_pct
            ) - current_gross_exposure

            capped_size = min(capped_size, max(remaining, 0))

        ###################################################
        # MIN FILTER
        ###################################################

        if capped_size < portfolio_value * self.config.min_position_size:
            return 0.0

        return round(float(capped_size), 2)
