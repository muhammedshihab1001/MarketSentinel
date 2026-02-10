from dataclasses import dataclass
import math


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

@dataclass(frozen=True)
class RiskConfig:
    """
    Institutional risk parameters.

    Frozen to prevent runtime mutation.
    """

    max_position_size: float = 0.10
    min_position_size: float = 0.01
    volatility_target: float = 0.02
    confidence_boost: float = 1.5


# ---------------------------------------------------
# POSITION SIZER
# ---------------------------------------------------

class PositionSizer:
    """
    Capital deployment engine.

    Guarantees:
    - bounded leverage
    - NaN safety
    - volatility clamps
    - confidence clamps
    - hard institutional ceilings
    """

    # HARD SAFETY LIMIT — cannot be overridden by config
    ABSOLUTE_MAX_POSITION = 0.20

    MIN_DEPLOYABLE_CAPITAL = 5_000

    VOL_FLOOR = 0.005
    VOL_CEILING = 0.10

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    # ---------------------------------------------------

    @staticmethod
    def _safe(v, fallback):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return fallback
        return float(v)

    # ---------------------------------------------------

    def size_position(
        self,
        signal: str,
        confidence: float,
        volatility: float | None,
        portfolio_value: float
    ) -> float:
        """
        Returns position size in dollars.
        """

        # ---------------------------------------
        # SIGNAL VALIDATION
        # ---------------------------------------

        if signal not in {"BUY", "SELL"}:
            return 0.0

        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive")

        if portfolio_value < self.MIN_DEPLOYABLE_CAPITAL:
            return 0.0

        # ---------------------------------------
        # SANITIZE INPUTS
        # ---------------------------------------

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

        # ---------------------------------------
        # VOL SCALING
        # ---------------------------------------

        vol_scalar = min(
            self.config.volatility_target / volatility,
            2.0
        )

        # ---------------------------------------
        # CONFIDENCE SCALING
        # ---------------------------------------

        confidence_scalar = 1 + (
            confidence *
            (self.config.confidence_boost - 1)
        )

        # ---------------------------------------
        # RAW SIZE
        # ---------------------------------------

        raw_size = (
            portfolio_value *
            self.config.max_position_size *
            vol_scalar *
            confidence_scalar
        )

        # ---------------------------------------
        # HARD CEILINGS
        # ---------------------------------------

        config_cap = portfolio_value * self.config.max_position_size
        absolute_cap = portfolio_value * self.ABSOLUTE_MAX_POSITION

        capped_size = min(raw_size, config_cap, absolute_cap)

        # ---------------------------------------
        # MICRO TRADE FILTER
        # ---------------------------------------

        if capped_size < portfolio_value * self.config.min_position_size:
            return 0.0

        return round(capped_size, 2)
