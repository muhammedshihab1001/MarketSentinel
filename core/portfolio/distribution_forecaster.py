import numpy as np
import logging

logger = logging.getLogger("marketsentinel.distribution")


class DistributionForecaster:
    """
    Converts forecasts into a probabilistic price surface.

    Designed for expected-value trading systems.
    """

    MIN_SIGMA = 1e-6
    MAX_SIGMA = 0.25

    def __init__(self, sigma_multiplier=1.25):
        self.sigma_multiplier = sigma_multiplier

    ########################################################

    def from_volatility(
        self,
        current_price: float,
        volatility: float,
        prob_up: float
    ):
        """
        Safe fallback distribution.
        """

        volatility = float(np.clip(volatility, self.MIN_SIGMA, self.MAX_SIGMA))
        prob_up = float(np.clip(prob_up, 0.01, 0.99))

        move = current_price * volatility * self.sigma_multiplier

        p50 = current_price + move * (prob_up - 0.5)
        p90 = current_price + move
        p10 = current_price - move

        return self._validate_surface(p10, p50, p90)

    ########################################################

    def from_lstm(
        self,
        current_price: float,
        forecast_prices: list
    ):
        """
        Builds percentile surface from LSTM forward path.
        """

        if forecast_prices is None or len(forecast_prices) < 5:
            raise RuntimeError("LSTM forecast insufficient.")

        arr = np.asarray(forecast_prices, dtype=np.float64)

        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite LSTM forecast.")

        p10 = float(np.percentile(arr, 10))
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))

        return self._validate_surface(p10, p50, p90)

    ########################################################

    def _validate_surface(self, p10, p50, p90):

        if not (p10 < p50 < p90):
            raise RuntimeError("Invalid forecast surface ordering.")

        width = (p90 - p10) / max(p50, 1e-6)

        if width > 0.8:
            logger.warning("Forecast distribution extremely wide.")

        return {
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90)
        }
