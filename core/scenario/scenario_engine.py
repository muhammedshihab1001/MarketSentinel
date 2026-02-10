import math


class ScenarioEngine:
    """
    Stress-tests forecast distributions.

    Guarantees:
    - no negative price space
    - numeric safety
    """

    MIN_PRICE = 0.01

    def __init__(self, volatility_multiplier=1.5):
        self.vol_multiplier = volatility_multiplier

    def _safe(self, v, fallback=0.0):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return fallback
        return float(v)

    def _clamp_price(self, v):
        return max(v, self.MIN_PRICE)

    # --------------------------------------------------

    def bull_case(self, mean, std):
        return self._clamp_price(mean + (2 * std))

    def bear_case(self, mean, std):
        return self._clamp_price(mean - (2 * std))

    def volatility_shock(self, mean, std):

        shock_std = std * self.vol_multiplier

        lower = self._clamp_price(mean - (3 * shock_std))
        upper = self._clamp_price(mean + (3 * shock_std))

        return lower, upper

    def sentiment_crash(self, mean):
        return self._clamp_price(mean * 0.9)

    # --------------------------------------------------

    def generate(self, forecast_distribution):

        mean = self._safe(
            forecast_distribution.get("mean_forecast")
        )

        std = abs(self._safe(
            forecast_distribution.get("std_dev")
        ))

        return {
            "bull_case": self.bull_case(mean, std),
            "bear_case": self.bear_case(mean, std),
            "volatility_shock_range": self.volatility_shock(mean, std),
            "sentiment_crash": self.sentiment_crash(mean)
        }
