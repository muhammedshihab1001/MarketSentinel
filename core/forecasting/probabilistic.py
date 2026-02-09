import numpy as np


class ForecastDistribution:
    """
    Builds probabilistic forecasts from model outputs.
    """

    def from_lstm_samples(self, predictions):

        predictions = np.array(predictions)

        mean = predictions.mean()
        std = predictions.std()

        lower = np.percentile(predictions, 5)
        upper = np.percentile(predictions, 95)

        return {
            "mean_forecast": float(mean),
            "std_dev": float(std),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "confidence_range": f"{lower:.2f} - {upper:.2f}"
        }

    def from_prophet(self, forecast_dict):

        return {
            "trend": forecast_dict["trend"],
            "lower_bound": forecast_dict["lower"],
            "upper_bound": forecast_dict["upper"],
            "confidence_range": f"{forecast_dict['lower']:.2f} - {forecast_dict['upper']:.2f}"
        }

    def combine(self, lstm_dist, prophet_dist):
        """
        Simple ensemble uncertainty fusion.
        """

        lower = min(lstm_dist["lower_bound"], prophet_dist["lower_bound"])
        upper = max(lstm_dist["upper_bound"], prophet_dist["upper_bound"])

        return {
            "ensemble_lower": lower,
            "ensemble_upper": upper,
            "ensemble_range": f"{lower:.2f} - {upper:.2f}"
        }
