import numpy as np
import pandas as pd
import os
import json
import logging
from typing import Dict, Any

from app.monitoring.metrics import DRIFT_DETECTED
from core.schema.feature_schema import get_schema_signature


logger = logging.getLogger("marketsentinel.drift")


class DriftDetector:
    """
    Production Drift Sentinel.

    Guarantees:
    - schema-bound baseline
    - non-crashing inference checks
    - global drift emission
    - variance safety
    """

    BASELINE_PATH = "artifacts/drift/baseline.json"
    BASELINE_VERSION = "2.0"

    MIN_SAMPLE_BASELINE = 50
    MIN_SAMPLE_INFERENCE = 20

    VARIANCE_RATIO_UPPER = 2.5
    VARIANCE_RATIO_LOWER = 0.4

    EPSILON = 1e-6

    def __init__(self, z_threshold: float = 3.0):

        self.z_threshold = z_threshold

        os.makedirs("artifacts/drift", exist_ok=True)

    # --------------------------------------------------
    # BASELINE CREATION
    # --------------------------------------------------

    def create_baseline(self, dataset: pd.DataFrame):

        if dataset.empty:
            raise ValueError("Cannot create drift baseline from empty dataset.")

        numeric = dataset.select_dtypes(include="number")

        if numeric.empty:
            raise RuntimeError("No numeric features available.")

        baseline: Dict[str, Any] = {
            "meta": {
                "baseline_version": self.BASELINE_VERSION,
                "schema_signature": get_schema_signature()
            },
            "features": {}
        }

        for col in numeric.columns:

            series = numeric[col].dropna()

            if len(series) < self.MIN_SAMPLE_BASELINE:
                raise RuntimeError(
                    f"Baseline unsafe: '{col}' insufficient samples."
                )

            std = series.std()

            if std < self.EPSILON:
                raise RuntimeError(
                    f"Baseline unsafe: '{col}' near-zero variance."
                )

            baseline["features"][col] = {
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

        logger.info("Drift baseline created.")

    # --------------------------------------------------

    def _load_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise RuntimeError("Drift baseline missing.")

        with open(self.BASELINE_PATH, "r") as f:
            baseline = json.load(f)

        meta = baseline.get("meta", {})

        if meta.get("schema_signature") != get_schema_signature():
            raise RuntimeError(
                "Baseline schema mismatch. Retraining required."
            )

        return baseline["features"]

    # --------------------------------------------------
    # DETECT DRIFT
    # --------------------------------------------------

    def detect(self, dataset: pd.DataFrame):

        try:

            if dataset.empty:
                logger.warning("Drift skipped — empty dataset.")
                return {"drift_detected": False, "details": {}}

            baseline = self._load_baseline()

            numeric = dataset.select_dtypes(include="number")

            baseline_features = set(baseline.keys())
            incoming_features = set(numeric.columns)

            if baseline_features != incoming_features:
                logger.critical(
                    "SCHEMA DRIFT DETECTED — blocking confidence."
                )

                DRIFT_DETECTED.set(1)

                return {
                    "drift_detected": True,
                    "reason": "schema_mismatch"
                }

            drift_detected = False
            drift_report = {}

            for col, stats in baseline.items():

                current = numeric[col].dropna()

                if len(current) < self.MIN_SAMPLE_INFERENCE:
                    logger.warning(
                        f"Drift skipped for {col} — insufficient sample."
                    )
                    continue

                mean_now = current.mean()
                var_now = current.var()

                std = max(stats["std"], self.EPSILON)
                baseline_var = max(stats["variance"], self.EPSILON)

                z_score = abs(mean_now - stats["mean"]) / std
                variance_ratio = var_now / baseline_var

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

                    logger.warning(
                        f"Drift detected in feature '{col}' | "
                        f"z={z_score:.2f} var_ratio={variance_ratio:.2f}"
                    )

                drift_report[col] = {
                    "mean_shift": bool(mean_shift),
                    "variance_shift": bool(variance_shift),
                    "min_breach": bool(min_breach),
                    "max_breach": bool(max_breach),
                    "z_score": float(z_score),
                    "variance_ratio": float(variance_ratio)
                }

            DRIFT_DETECTED.set(1 if drift_detected else 0)

            return {
                "drift_detected": drift_detected,
                "details": drift_report
            }

        except Exception:

            logger.exception(
                "Drift detector failure — forcing alert."
            )

            DRIFT_DETECTED.set(1)

            return {
                "drift_detected": True,
                "reason": "detector_failure"
            }
