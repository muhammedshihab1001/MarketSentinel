import math


class ScenarioEngine:
    """
    Institutional Scenario Engine.

    Guarantees:
    ✔ no negative price space
    ✔ volatility floor
    ✔ explosion guard
    ✔ deterministic output
    ✔ distribution validation
    ✔ numeric safety
    """

    MIN_PRICE = 0.01
    MIN_STD = 1e-4

    # Prevent insane forecasts
    MAX_STD_RATIO = 3.5     # std cannot exceed 350% of mean
    MAX_PRICE_MULTIPLIER = 5.0

    def __init__(self, volatility_multiplier=1.5):
        self.vol_multiplier = volatility_multiplier

    ##################################################
    # SAFETY
    ##################################################

    def _safe(self, v, fallback=0.0):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return fallback
        return float(v)

    def _clamp_price(self, v):
        return max(float(v), self.MIN_PRICE)

    ##################################################
    # DISTRIBUTION VALIDATION (VERY IMPORTANT)
    ##################################################

    def _validate_distribution(self, mean, std):

        if not math.isfinite(mean):
            raise RuntimeError("Forecast mean is non-finite.")

        if not math.isfinite(std):
            raise RuntimeError("Forecast std is non-finite.")

        std = max(abs(std), self.MIN_STD)

        # Prevent volatility explosion
        if std > abs(mean) * self.MAX_STD_RATIO:
            std = abs(mean) * self.MAX_STD_RATIO

        return mean, std

    ##################################################
    # SCENARIOS
    ##################################################

    def bull_case(self, mean, std):

        price = mean + (2 * std)

        # Prevent unrealistic upside
        price = min(price, mean * self.MAX_PRICE_MULTIPLIER)

        return self._clamp_price(price)

    def bear_case(self, mean, std):

        price = mean - (2 * std)

        return self._clamp_price(price)

    def volatility_shock(self, mean, std):

        shock_std = std * self.vol_multiplier

        lower = self._clamp_price(mean - (3 * shock_std))
        upper = self._clamp_price(
            min(mean + (3 * shock_std),
                mean * self.MAX_PRICE_MULTIPLIER)
        )

        return lower, upper

    def sentiment_crash(self, mean):

        crash = mean * 0.9

        return self._clamp_price(crash)

    ##################################################
    # PUBLIC API
    ##################################################

    def generate(self, forecast_distribution):

        if not isinstance(forecast_distribution, dict):
            raise RuntimeError(
                "Forecast distribution must be a dictionary."
            )

        mean = self._safe(
            forecast_distribution.get("mean_forecast")
        )

        std = self._safe(
            forecast_distribution.get("std_dev")
        )

        mean, std = self._validate_distribution(mean, std)

        return {
            "bull_case": self.bull_case(mean, std),
            "bear_case": self.bear_case(mean, std),
            "volatility_shock_range": self.volatility_shock(mean, std),
            "sentiment_crash": self.sentiment_crash(mean)
        }
