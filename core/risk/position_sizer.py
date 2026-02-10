from dataclasses import dataclass


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

@dataclass
class RiskConfig:
    """
    Institutional risk parameters.

    Conservative defaults.
    Safe for live deployment.
    """

    max_position_size: float = 0.10     # 10% of portfolio
    min_position_size: float = 0.01     # avoid micro trades
    volatility_target: float = 0.02     # ideal daily vol
    confidence_boost: float = 1.5       # amplify strong signals


# ---------------------------------------------------
# POSITION SIZER
# ---------------------------------------------------

class PositionSizer:
    """
    Determines how much capital to deploy.

    Philosophy:
        risk first
        profit second
    """

    def __init__(self, config: RiskConfig = RiskConfig()):
        self.config = config

    # ---------------------------------------------------

    def size_position(
        self,
        signal: str,
        confidence: float,
        volatility: float | None,
        portfolio_value: float
    ):
        """
        Returns position size in dollars.
        """

        if signal == "HOLD":
            return 0.0

        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive")

        # fallback volatility
        if volatility is None or volatility == 0:
            volatility = self.config.volatility_target

        # ---------------------------------------
        # Volatility scaling
        # Lower vol → larger position
        # Higher vol → smaller position
        # ---------------------------------------

        vol_scalar = min(
            self.config.volatility_target / volatility,
            2.0  # never double size
        )

        # ---------------------------------------
        # Confidence scaling
        # ---------------------------------------

        confidence_scalar = 1 + (
            confidence * (self.config.confidence_boost - 1)
        )

        raw_size = (
            portfolio_value *
            self.config.max_position_size *
            vol_scalar *
            confidence_scalar
        )

        # Hard cap
        capped_size = min(
            raw_size,
            portfolio_value * self.config.max_position_size
        )

        # Avoid tiny trades
        if capped_size < portfolio_value * self.config.min_position_size:
            return 0.0

        return round(capped_size, 2)
