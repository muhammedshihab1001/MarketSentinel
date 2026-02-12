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
