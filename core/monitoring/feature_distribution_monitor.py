import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path


logger = logging.getLogger(__name__)


############################################################
# CONFIG
############################################################

DRIFT_THRESHOLD_MEAN = 3.0        # Z-score shift threshold
DRIFT_THRESHOLD_STD = 2.0
PSI_THRESHOLD_WARNING = 0.2
PSI_THRESHOLD_CRITICAL = 0.5

EPS = 1e-8


############################################################
# PSI CALCULATION
############################################################

def _calculate_psi(expected: np.ndarray,
                   actual: np.ndarray,
                   bins: int = 10) -> float:

    breakpoints = np.linspace(
        0, 100, bins + 1
    )

    expected_perc = np.percentile(expected, breakpoints)
    actual_perc = np.percentile(actual, breakpoints)

    expected_counts, _ = np.histogram(expected, bins=expected_perc)
    actual_counts, _ = np.histogram(actual, bins=expected_perc)

    expected_ratio = expected_counts / (len(expected) + EPS)
    actual_ratio = actual_counts / (len(actual) + EPS)

    psi = np.sum(
        (expected_ratio - actual_ratio) *
        np.log((expected_ratio + EPS) / (actual_ratio + EPS))
    )

    return float(psi)


############################################################
# BASELINE BUILDER
############################################################

class FeatureDistributionMonitor:

    def __init__(self, baseline_path: str):
        self.baseline_path = Path(baseline_path)
        self.baseline: Dict[str, Any] = {}

    ########################################################
    # BUILD BASELINE FROM TRAINING DATA
    ########################################################

    def build_baseline(self, df: pd.DataFrame):

        baseline = {}

        for col in df.columns:
            values = df[col].dropna().values

            baseline[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p01": float(np.percentile(values, 1)),
                "p99": float(np.percentile(values, 99)),
            }

        self.baseline = baseline

        self._save()

        logger.info("Feature baseline distribution saved.")

    ########################################################
    # LOAD BASELINE
    ########################################################

    def load(self):

        if not self.baseline_path.exists():
            raise RuntimeError("Baseline distribution not found.")

        with open(self.baseline_path, "r") as f:
            self.baseline = json.load(f)

    ########################################################
    # SAVE BASELINE
    ########################################################

    def _save(self):

        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.baseline_path, "w") as f:
            json.dump(self.baseline, f, indent=2)

    ########################################################
    # COMPARE INFERENCE BATCH
    ########################################################

    def compare(self,
                df: pd.DataFrame,
                hard_fail: bool = False) -> Dict[str, Any]:

        if not self.baseline:
            self.load()

        report = {}
        critical_drift = False

        for col in df.columns:

            if col not in self.baseline:
                raise RuntimeError(f"Feature {col} missing in baseline.")

            current = df[col].dropna().values
            base = self.baseline[col]

            mean_shift = abs(
                (np.mean(current) - base["mean"]) /
                (base["std"] + EPS)
            )

            std_ratio = (
                np.std(current) /
                (base["std"] + EPS)
            )

            psi = _calculate_psi(
                expected=np.random.normal(
                    base["mean"],
                    base["std"] + EPS,
                    size=len(current)
                ),
                actual=current
            )

            drift_flag = False

            if mean_shift > DRIFT_THRESHOLD_MEAN:
                drift_flag = True

            if std_ratio > DRIFT_THRESHOLD_STD:
                drift_flag = True

            if psi > PSI_THRESHOLD_CRITICAL:
                drift_flag = True

            if drift_flag:
                critical_drift = True

            report[col] = {
                "mean_shift_z": float(mean_shift),
                "std_ratio": float(std_ratio),
                "psi": float(psi),
                "drift": drift_flag
            }

        if critical_drift:
            msg = "Critical feature distribution drift detected."

            if hard_fail:
                raise RuntimeError(msg)

            logger.warning(msg)

        return report
