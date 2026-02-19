import math


class ExpectedValueEngine:
    """
    Institutional Expected Value calculator (Return-Space Corrected).

    EV = p * expected_up_return + (1 - p) * expected_down_return

    All inputs are treated as RETURNS (not prices).
    """

    MIN_EDGE = 0.0008  # 0.08% minimum edge (was too strict)

    ###################################################

    @staticmethod
    def compute(
        prob_up: float,
        current_price: float,   # kept for compatibility (not used)
        forecast_up: float,     # expected positive return (e.g. +0.02)
        forecast_down: float    # expected negative return (e.g. -0.015)
    ) -> float:

        # Basic numeric safety
        if not all(map(math.isfinite, [
            prob_up,
            forecast_up,
            forecast_down
        ])):
            return 0.0

        # Probability bounds
        prob_up = max(0.0, min(prob_up, 1.0))

        ###################################################
        # EV IN RETURN SPACE
        ###################################################

        ev = (
            prob_up * forecast_up
            + (1.0 - prob_up) * forecast_down
        )

        # Clamp extreme nonsense
        if abs(ev) > 0.20:
            return 0.0

        return ev

    ###################################################

    @classmethod
    def worthy_trade(cls, ev: float) -> bool:
        return math.isfinite(ev) and abs(ev) > cls.MIN_EDGE
