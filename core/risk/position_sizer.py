from dataclasses import dataclass
import math
import os
import numpy as np


@dataclass(frozen=True)
class RiskConfig:
    """
    Immutable production risk parameters.
    """

    max_position_size: float = 0.08
    min_position_size: float = 0.01

    volatility_target: float = 0.02

    confidence_boost: float = 1.35

    max_gross_exposure_pct: float = 0.40

    max_single_trade_dollars: float = 25_000


class PositionSizer:
    """
    Institutional volatility-aware capital allocator.

    Guarantees:
    ✔ no leverage
    ✔ no inverse-vol explosion
    ✔ env-safe drawdown handling
    ✔ numeric stability
    """

    ABSOLUTE_MAX_POSITION = 0.15
    MIN_DEPLOYABLE_CAPITAL = 10_000

    VOL_FLOOR = 0.006
    VOL_CEILING = 0.12

    MAX_VOL_SCALAR = 1.5   # <- tightened (important)

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    ###################################################
    # NUMERIC SAFETY
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
    # DRAW DOWN SCALAR (ENV SAFE)
    ###################################################

    def _drawdown_scalar(self):

        raw = os.getenv("PORTFOLIO_DRAWDOWN_PCT", "0")

        try:
            drawdown = float(raw)
        except Exception:
            return 1.0

        # normalize if someone passes 5 instead of 0.05
        if drawdown > 1:
            drawdown /= 100

        drawdown = abs(drawdown)
        drawdown = min(drawdown, 0.80)

        if drawdown < 0.05:
            return 1.0
        if drawdown < 0.10:
            return 0.75
        if drawdown < 0.20:
            return 0.50

        return 0.25

    ###################################################
    # POSITION SIZING
    ###################################################

    def size_position(
        self,
        signal: str,
        confidence: float,
        volatility: float | None,
        portfolio_value: float,
        current_gross_exposure: float | None = None
    ) -> float:

        # 🔴 HARD SIGNAL CONTRACT
        if signal != "BUY":
            return 0.0

        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive")

        if portfolio_value < self.MIN_DEPLOYABLE_CAPITAL:
            return 0.0

        confidence = self._safe(confidence, 0.5)
        confidence = max(0.0, min(confidence, 1.0))

        volatility = self._safe(
            volatility,
            self.config.volatility_target
        )

        volatility = max(
            self.VOL_FLOOR,
            min(volatility, self.VOL_CEILING)
        )

        ###################################################
        # SCALARS
        ###################################################

        confidence_scalar = math.tanh(
            confidence * self.config.confidence_boost
        )

        confidence_scalar = max(confidence_scalar, 0.25)

        vol_scalar = math.sqrt(
            self.config.volatility_target / volatility
        )

        vol_scalar = min(vol_scalar, self.MAX_VOL_SCALAR)

        dd_scalar = self._drawdown_scalar()

        ###################################################
        # RAW SIZE
        ###################################################

        raw_size = (
            portfolio_value *
            self.config.max_position_size *
            vol_scalar *
            confidence_scalar *
            dd_scalar
        )

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
        # GROSS EXPOSURE GUARD
        ###################################################

        if current_gross_exposure is not None:

            current_gross_exposure = max(current_gross_exposure, 0)

            remaining_capacity = (
                portfolio_value *
                self.config.max_gross_exposure_pct
            ) - current_gross_exposure

            remaining_capacity = max(remaining_capacity, 0)

            capped_size = min(capped_size, remaining_capacity)

        ###################################################
        # MIN SIZE FILTER
        ###################################################

        if capped_size < portfolio_value * self.config.min_position_size:
            return 0.0

        return round(float(capped_size), 2)
