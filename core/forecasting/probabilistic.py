import numpy as np


class ForecastDistribution:
    """
    Institutional probabilistic forecast builder.

    Guarantees:
    ✔ finite outputs
    ✔ variance floor
    ✔ tail protection
    ✔ numeric stability
    ✔ minimum sample enforcement
    """

    MIN_SAMPLES = 20
    VAR_FLOOR = 1e-4

    WINSOR_LO = 1
    WINSOR_HI = 99

    ###################################################

    def _sanitize(self, predictions):

        if predictions is None:
            raise RuntimeError("Forecast predictions missing.")

        arr = np.asarray(predictions, dtype=np.float64)

        if arr.size < self.MIN_SAMPLES:
            raise RuntimeError(
                f"Forecast sample too small ({arr.size})."
            )

        finite_mask = np.isfinite(arr)

        if not finite_mask.any():
            raise RuntimeError("All forecast samples invalid.")

        arr = arr[finite_mask]

        if arr.size < self.MIN_SAMPLES:
            raise RuntimeError(
                "Too many invalid forecast samples."
            )

        return arr

    ###################################################

    def _winsorize(self, arr):

        lo = np.percentile(arr, self.WINSOR_LO)
        hi = np.percentile(arr, self.WINSOR_HI)

        return np.clip(arr, lo, hi)

    ###################################################

    def from_lstm_samples(self, predictions):

        arr = self._sanitize(predictions)

        arr = self._winsorize(arr)

        mean = float(arr.mean())

        std = float(max(arr.std(), self.VAR_FLOOR))

        lower = float(np.percentile(arr, 5))
        upper = float(np.percentile(arr, 95))

        if lower > upper:
            raise RuntimeError("Forecast bounds corrupted.")

        return {
            "mean_forecast": mean,
            "std_dev": std,
            "lower_bound": lower,
            "upper_bound": upper,
            "confidence_range": f"{lower:.4f} - {upper:.4f}",
            "sample_size": int(arr.size)
        }
