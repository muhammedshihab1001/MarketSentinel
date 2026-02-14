from dataclasses import dataclass
import math
import os
import numpy as np


###################################################
# SAFE ENV LOADER (CONFIG POISONING DEFENSE)
###################################################

def _env_float(key: str, default: float) -> float:
    try:
        v = float(os.getenv(key, str(default)))

        if not math.isfinite(v):
            return default

        # Prevent malicious / broken configs
        if abs(v) > 10:
            return default

        return v

    except Exception:
        return default


###################################################
# RISK CONFIG
###################################################

@dataclass(frozen=True)
class RiskConfig:

    max_position_size: float
    min_position_size: float

    volatility_target: float
    confidence_boost: float

    max_gross_exposure_pct: float
    max_single_trade_dollars: float

    fractional_kelly: float
    volatility_shock_level: float
    correlation_heat_cap: float

    max_drawdown_kill: float

    ###################################################

    @staticmethod
    def load():

        cfg = RiskConfig(
            max_position_size=_env_float("MAX_POSITION_SIZE", 0.07),
            min_position_size=_env_float("MIN_POSITION_SIZE", 0.01),

            volatility_target=_env_float("VOL_TARGET", 0.02),
            confidence_boost=_env_float("CONFIDENCE_BOOST", 1.25),

            max_gross_exposure_pct=_env_float("MAX_GROSS_EXPOSURE", 0.35),
            max_single_trade_dollars=_env_float("MAX_SINGLE_TRADE", 20_000),

            fractional_kelly=_env_float("FRACTIONAL_KELLY", 0.20),
            volatility_shock_level=_env_float("VOL_SHOCK_LEVEL", 0.07),
            correlation_heat_cap=_env_float("HEAT_CAP", 0.20),

            max_drawdown_kill=_env_float("MAX_DRAWDOWN_KILL", 0.30),
        )

        ###################################################
        # HARD VALIDATION
        ###################################################

        if not (0 < cfg.max_position_size <= 0.20):
            raise RuntimeError("Unsafe max_position_size")

        if not (0 < cfg.max_gross_exposure_pct <= 1.5):
            raise RuntimeError("Unsafe gross exposure")

        if not (0 <= cfg.max_drawdown_kill <= 0.8):
            raise RuntimeError("Unsafe drawdown kill")

        if cfg.volatility_target <= 0:
            raise RuntimeError("Invalid volatility target")

        return cfg


###################################################
# POSITION SIZER (INSTITUTIONAL)
###################################################

class PositionSizer:

    ABSOLUTE_MAX_POSITION = 0.15
    MIN_DEPLOYABLE_CAPITAL = 10_000

    VOL_FLOOR = 0.007
    VOL_CEILING = 0.12
    EFFECTIVE_VOL_FLOOR = 0.012

    MAX_VOL_SCALAR = 1.20
    MIN_CONFIDENCE_FLOOR = 0.25

    MAX_KELLY_FRACTION = 0.06

    PORTFOLIO_VOL_SOFT_CAP = 0.035
    PORTFOLIO_VOL_HARD_CAP = 0.06

    ###################################################

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig.load()

    ###################################################
    # SAFE FLOAT
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
    # CONVEX DRAWDOWN SCALING
    ###################################################

    def _drawdown_scalar(self, drawdown):

        if drawdown < 0.05:
            return 1.0

        if drawdown < 0.10:
            return 0.85

        if drawdown < 0.15:
            return 0.70

        if drawdown < 0.20:
            return 0.50

        if drawdown < self.config.max_drawdown_kill:
            return 0.30

        return 0.0

    ###################################################
    # PORTFOLIO VOL THROTTLE
    ###################################################

    def _portfolio_vol_scalar(self):

        raw = os.getenv("PORTFOLIO_VOLATILITY", "0.02")

        try:
            vol = float(raw)
        except Exception:
            return 1.0

        if vol <= self.PORTFOLIO_VOL_SOFT_CAP:
            return 1.0

        if vol >= self.PORTFOLIO_VOL_HARD_CAP:
            return 0.25

        return max(
            0.25,
            self.PORTFOLIO_VOL_SOFT_CAP / vol
        )

    ###################################################
    # VOL SHOCK GUARD
    ###################################################

    def _volatility_shock_scalar(self, volatility):

        if volatility <= self.config.volatility_shock_level:
            return 1.0

        ratio = self.config.volatility_shock_level / volatility

        return max(0.25, ratio)

    ###################################################
    # STABILIZED KELLY
    ###################################################

    def _kelly_cap(self, confidence, volatility, portfolio_value):

        edge = max(confidence - 0.5, 0)

        if edge <= 0:
            return 0

        effective_vol = max(volatility, self.EFFECTIVE_VOL_FLOOR)

        variance = effective_vol ** 2

        kelly_fraction = edge / variance
        kelly_fraction *= self.config.fractional_kelly

        kelly_fraction = min(
            kelly_fraction,
            self.MAX_KELLY_FRACTION
        )

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
        current_heat: float | None = None
    ) -> float:

        if signal != "BUY":
            return 0.0

        portfolio_value = self._safe(portfolio_value, 0)

        if portfolio_value < self.MIN_DEPLOYABLE_CAPITAL:
            return 0.0

        ###################################################
        # DRAWDOWN CONTROL
        ###################################################

        drawdown = float(os.getenv("PORTFOLIO_DRAWDOWN_PCT", "0"))

        if drawdown > 1:
            drawdown /= 100

        drawdown = min(max(drawdown, 0), 0.95)

        dd_scalar = self._drawdown_scalar(drawdown)

        if dd_scalar == 0:
            return 0.0

        ###################################################
        # PORTFOLIO VOL CONTROL
        ###################################################

        port_vol_scalar = self._portfolio_vol_scalar()

        ###################################################
        # INPUT SANITY
        ###################################################

        confidence = max(
            self._safe(confidence, 0.5),
            self.MIN_CONFIDENCE_FLOOR
        )

        confidence = min(confidence, 0.92)

        volatility = self._safe(
            volatility,
            self.config.volatility_target
        )

        volatility = max(
            self.VOL_FLOOR,
            min(volatility, self.VOL_CEILING)
        )

        effective_vol = max(volatility, self.EFFECTIVE_VOL_FLOOR)

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

        confidence_scalar = confidence ** 1.3

        vol_scalar = math.sqrt(
            self.config.volatility_target / effective_vol
        )

        vol_scalar = min(vol_scalar, self.MAX_VOL_SCALAR)

        shock_scalar = self._volatility_shock_scalar(volatility)

        ###################################################
        # RAW SIZE
        ###################################################

        raw_size = (
            portfolio_value *
            self.config.max_position_size *
            confidence_scalar *
            vol_scalar *
            shock_scalar *
            dd_scalar *
            port_vol_scalar
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

        if capped_size > portfolio_value:
            return 0.0

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
