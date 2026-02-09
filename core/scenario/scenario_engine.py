import numpy as np


class ScenarioEngine:
    """
    Generates alternative future scenarios
    to stress-test forecasts.
    """

    def __init__(self, volatility_multiplier=1.5):
        self.vol_multiplier = volatility_multiplier

    def bull_case(self, forecast_mean, std):
        """
        Strong upside scenario.
        """
        return forecast_mean + (2 * std)

    def bear_case(self, forecast_mean, std):
        """
        Strong downside scenario.
        """
        return forecast_mean - (2 * std)

    def volatility_shock(self, forecast_mean, std):
        """
        Expands uncertainty dramatically.
        """
        shock_std = std * self.vol_multiplier

        lower = forecast_mean - (3 * shock_std)
        upper = forecast_mean + (3 * shock_std)

        return lower, upper

    def sentiment_crash(self, forecast_mean):
        """
        Models sudden negative narrative shift.
        """
        return forecast_mean * 0.9

    def generate(self, forecast_distribution):
        """
        Main scenario generator.
        """

        mean = forecast_distribution["mean_forecast"]
        std = forecast_distribution["std_dev"]

        bull = self.bull_case(mean, std)
        bear = self.bear_case(mean, std)
        shock_low, shock_high = self.volatility_shock(mean, std)
        sentiment_drop = self.sentiment_crash(mean)

        return {
            "bull_case": bull,
            "bear_case": bear,
            "volatility_shock_range": (shock_low, shock_high),
            "sentiment_crash": sentiment_drop
        }
