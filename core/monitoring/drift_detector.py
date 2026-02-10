import numpy as np
import pandas as pd
import os
import json
from typing import Dict, Any

from app.monitoring.metrics import DRIFT_EVENTS_COUNTER


class DriftDetector:
    """
    Institutional Feature Drift Detector.

    Detects:
    - mean shift (z-score)
    - variance shift
    - min/max breach
    - schema mismatch

    Emits Prometheus alerts.

    Designed as a production safety system — NOT a research tool.
    """

    BASELINE_PATH = "artifacts/drift/baseline.json"

    MIN_SAMPLE_BASELINE = 50
    MIN_SAMPLE_INFERENCE = 20

    VARIANCE_RATIO_UPPER = 2.5
    VARIANCE_RATIO_LOWER = 0.4

    def __init__(self, z_threshold: float = 3.0):

        self.z_threshold = z_threshold

        os.makedirs("artifacts/drift", exist_ok=True)

    # --------------------------------------------------
    # BASELINE CREATION
    # MUST RUN DURING TRAINING PIPELINE
    # --------------------------------------------------

    def create_baseline(self, dataset: pd.DataFrame):

        if dataset.empty:
            raise ValueError("Cannot create drift baseline from empty dataset.")

        numeric = dataset.select_dtypes(include="number")

        if numeric.empty:
            raise RuntimeError("No numeric features available for drift baseline.")

        baseline: Dict[str, Any] = {}

        for col in numeric.columns:

            series = numeric[col].dropna()

            if len(series) < self.MIN_SAMPLE_BASELINE:
                raise RuntimeError(
                    f"Baseline unsafe: feature '{col}' has insufficient samples."
                )

            std = series.std()

            if std == 0:
                raise RuntimeError(
                    f"Baseline unsafe: feature '{col}' has zero variance."
                )

            baseline[col] = {
                "mean": float(series.mean()),
                "std": float(std),
                "variance": float(series.var()),
                "min": float(series.min()),
                "max": float(series.max()),
                "count": int(len(series))
            }

        tmp_path = self.BASELINE_PATH + ".tmp"

        with open(tmp_path, "w") as f:
            json.dump(baseline, f, indent=4)

        os.replace(tmp_path, self.BASELINE_PATH)

    # --------------------------------------------------

    def _load_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise RuntimeError(
                "Drift baseline missing. Training pipeline must generate it."
            )

        with open(self.BASELINE_PATH, "r") as f:
            return json.load(f)

    # --------------------------------------------------
    # DETECT DRIFT
    # --------------------------------------------------

    def detect(self, dataset: pd.DataFrame):

        if dataset.empty:
            raise RuntimeError("Drift detection received empty dataset.")

        baseline = self._load_baseline()

        numeric = dataset.select_dtypes(include="number")

        baseline_features = set(baseline.keys())
        incoming_features = set(numeric.columns)

        if baseline_features != incoming_features:
            missing = baseline_features - incoming_features
            extra = incoming_features - baseline_features

            raise RuntimeError(
                f"Feature schema mismatch detected. "
                f"Missing={missing}, Extra={extra}"
            )

        drift_detected = False
        drift_report = {}

        for col, stats in baseline.items():

            current = numeric[col].dropna()

            if len(current) < self.MIN_SAMPLE_INFERENCE:
                raise RuntimeError(
                    f"Inference drift check unsafe: '{col}' sample too small."
                )

            mean_now = current.mean()
            var_now = current.var()

            z_score = abs(mean_now - stats["mean"]) / stats["std"]

            variance_ratio = var_now / stats["variance"]

            min_breach = current.min() < stats["min"]
            max_breach = current.max() > stats["max"]

            variance_shift = (
                variance_ratio > self.VARIANCE_RATIO_UPPER
                or variance_ratio < self.VARIANCE_RATIO_LOWER
            )

            mean_shift = z_score > self.z_threshold

            feature_drift = any([
                mean_shift,
                variance_shift,
                min_breach,
                max_breach
            ])

            if feature_drift:
                drift_detected = True

                DRIFT_EVENTS_COUNTER.labels(feature=col).inc()

            drift_report[col] = {
                "mean_shift": bool(mean_shift),
                "variance_shift": bool(variance_shift),
                "min_breach": bool(min_breach),
                "max_breach": bool(max_breach),
                "z_score": float(z_score),
                "variance_ratio": float(variance_ratio)
            }

        return {
            "drift_detected": drift_detected,
            "details": drift_report
        }
