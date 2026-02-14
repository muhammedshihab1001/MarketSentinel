import math


class ExpectedValueEngine:
    """
    Institutional Expected Value calculator.

    EV = (p * gain) - ((1-p) * loss)

    We normalize by price so edge becomes comparable
    across tickers.
    """

    MIN_EDGE = 0.002  # 0.2% required edge

    ###################################################

    @staticmethod
    def compute(
        prob_up: float,
        current_price: float,
        forecast_up: float,
        forecast_down: float
    ) -> float:

        if not all(map(math.isfinite, [
            prob_up,
            current_price,
            forecast_up,
            forecast_down
        ])):
            return -1.0

        if current_price <= 0:
            return -1.0

        gain = max(forecast_up - current_price, 0.0)
        loss = max(current_price - forecast_down, 0.0)

        # asymmetric protection
        if gain == 0 or loss == 0:
            return -1.0

        ev = (prob_up * gain) - ((1 - prob_up) * loss)

        return ev / current_price

    ###################################################

    @classmethod
    def worthy_trade(cls, ev: float) -> bool:
        return math.isfinite(ev) and ev > cls.MIN_EDGE
