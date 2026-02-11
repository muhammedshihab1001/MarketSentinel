from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class RiskConfig:
    """
    Production risk parameters.
    Immutable to prevent runtime mutation.
    """

    max_position_size: float = 0.08
    min_position_size: float = 0.01

    volatility_target: float = 0.02

    confidence_boost: float = 1.35

    max_gross_exposure_pct: float = 0.40

    max_single_trade_dollars: float = 25_000


class PositionSizer:
    """
    Volatility-aware capital deployment engine.
    """

    ABSOLUTE_MAX_POSITION = 0.15
    MIN_DEPLOYABLE_CAPITAL = 10_000

    VOL_FLOOR = 0.006
    VOL_CEILING = 0.12

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    @staticmethod
    def _safe(v, fallback):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return fallback
        return float(v)

    def _drawdown_scalar(self):

        try:
            drawdown = float(
                os.getenv("PORTFOLIO_DRAWDOWN_PCT", "0.0")
            )
        except Exception:
            drawdown = 0.0

        drawdown = abs(drawdown)

        # clamp to sane bounds
        drawdown = max(0.0, min(drawdown, 0.80))

        if drawdown < 0.05:
            return 1.0

        if drawdown < 0.10:
            return 0.75

        if drawdown < 0.20:
            return 0.50

        return 0.25

    def size_position(
        self,
        signal: str,
        confidence: float,
        volatility: float | None,
        portfolio_value: float,
        current_gross_exposure: float | None = None
    ) -> float:

        if signal not in {"BUY", "SELL"}:
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

        confidence_scalar = math.tanh(
            confidence * self.config.confidence_boost
        )

        confidence_scalar = max(confidence_scalar, 0.25)

        vol_scalar = math.sqrt(
            self.config.volatility_target / volatility
        )

        vol_scalar = min(vol_scalar, 1.75)

        dd_scalar = self._drawdown_scalar()

        raw_size = (
            portfolio_value *
            self.config.max_position_size *
            vol_scalar *
            confidence_scalar *
            dd_scalar
        )

        config_cap = portfolio_value * self.config.max_position_size
        absolute_cap = portfolio_value * self.ABSOLUTE_MAX_POSITION
        trade_cap = self.config.max_single_trade_dollars

        capped_size = min(
            raw_size,
            config_cap,
            absolute_cap,
            trade_cap
        )

        if current_gross_exposure is not None:

            remaining_capacity = (
                portfolio_value *
                self.config.max_gross_exposure_pct
            ) - current_gross_exposure

            if remaining_capacity <= 0:
                return 0.0

            capped_size = min(capped_size, remaining_capacity)

        if capped_size < portfolio_value * self.config.min_position_size:
            return 0.0

        return round(capped_size, 2)
