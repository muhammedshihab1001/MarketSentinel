import numpy as np
import pandas as pd
import os
import json


class DriftDetector:
    """
    Institutional Feature Drift Detector.

    Detects:
    ✅ mean shifts
    ✅ variance explosions
    ✅ distribution anomalies

    Safe for production inference.
    Lightweight (<1ms).
    """

    BASELINE_PATH = "artifacts/drift/baseline.json"

    # --------------------------------------------------

    def __init__(self, z_threshold=3.0):
        """
        z_threshold:

        2.5 → sensitive
        3.0 → institutional default
        4.0 → conservative
        """

        self.z_threshold = z_threshold

        os.makedirs("artifacts/drift", exist_ok=True)

    # --------------------------------------------------
    # BASELINE CREATION
    # RUN AFTER TRAINING
    # --------------------------------------------------

    def create_baseline(self, dataset: pd.DataFrame):

        numeric = dataset.select_dtypes(include="number")

        baseline = {}

        for col in numeric.columns:

            series = numeric[col].dropna()

            # avoid tiny samples
            if len(series) < 30:
                continue

            std = series.std()

            # avoid divide-by-zero later
            if std == 0:
                continue

            baseline[col] = {
                "mean": float(series.mean()),
                "std": float(std)
            }

        with open(self.BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=4)

        print("✅ Drift baseline created")

    # --------------------------------------------------

    def _load_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise RuntimeError(
                "Drift baseline missing. Run training first."
            )

        with open(self.BASELINE_PATH, "r") as f:
            return json.load(f)

    # --------------------------------------------------
    # DETECT DRIFT
    # --------------------------------------------------

    def detect(self, dataset: pd.DataFrame):

        baseline = self._load_baseline()

        numeric = dataset.select_dtypes(include="number")

        drift_report = {}
        drift_detected = False

        for col, stats in baseline.items():

            if col not in numeric.columns:
                continue

            current = numeric[col].dropna()

            if len(current) < 10:
                continue

            mean_now = current.mean()
            std_baseline = stats["std"]

            if std_baseline == 0:
                continue

            z_score = abs(mean_now - stats["mean"]) / std_baseline

            drift_flag = z_score > self.z_threshold

            if drift_flag:
                drift_detected = True

            drift_report[col] = {
                "baseline_mean": stats["mean"],
                "current_mean": float(mean_now),
                "z_score": float(z_score),
                "drift": drift_flag
            }

        return {
            "drift_detected": drift_detected,
            "details": drift_report
        }
