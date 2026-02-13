import numpy as np
import pandas as pd
import os
import json
import logging
import hashlib
from datetime import datetime

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES,
    DTYPE
)

from app.inference.model_loader import ModelLoader

logger = logging.getLogger("marketsentinel.drift")

try:
    from app.monitoring.metrics import DRIFT_DETECTED
except Exception:
    class _NoOpMetric:
        def set(self, *_):
            pass
    DRIFT_DETECTED = _NoOpMetric()


class DriftDetector:

    BASELINE_PATH = "artifacts/drift/baseline.json"
    BASELINE_VERSION = "11.0"

    MIN_SAMPLE_BASELINE = 100
    MIN_SAMPLE_INFERENCE = 30

    VARIANCE_RATIO_UPPER = 3.0
    VARIANCE_RATIO_LOWER = 0.30

    PSI_ALERT = 0.25
    PSI_CRITICAL = 0.40

    MAX_INFERENCE_ROWS = 500

    EPSILON = 1e-6
    MIN_ACTIVE_FEATURE_RATIO = 0.60

    ########################################################

    def __init__(self, z_threshold: float = 3.0):

        self.z_threshold = z_threshold

        os.makedirs("artifacts/drift", exist_ok=True)

        self.hard_fail = os.getenv(
            "DRIFT_HARD_FAIL",
            "true"
        ).lower() == "true"

    ########################################################
    # PATH SAFETY
    ########################################################

    def _safe_baseline_path(self):

        base = os.path.realpath("artifacts/drift")
        path = os.path.realpath(self.BASELINE_PATH)

        if not path.startswith(base):
            raise RuntimeError("Baseline path traversal detected.")

        return path

    ########################################################
    # PSI
    ########################################################

    def _psi(self, expected, actual, bins=10):

        expected = np.asarray(expected)
        actual = np.asarray(actual)

        quantiles = np.percentile(
            expected,
            np.linspace(0, 100, bins + 1)
        )

        quantiles = np.unique(quantiles)

        if len(quantiles) < 2:
            return 0.0

        expected_counts = np.histogram(expected, bins=quantiles)[0]
        actual_counts = np.histogram(actual, bins=quantiles)[0]

        expected_perc = expected_counts / max(len(expected), self.EPSILON)
        actual_perc = actual_counts / max(len(actual), self.EPSILON)

        psi = np.sum(
            (actual_perc - expected_perc) *
            np.log((actual_perc + self.EPSILON) /
                   (expected_perc + self.EPSILON))
        )

        return float(psi)

    ########################################################
    # BASELINE HASH
    ########################################################

    @staticmethod
    def _baseline_hash(payload: dict) -> str:

        clone = dict(payload)
        clone.pop("integrity_hash", None)

        canonical = json.dumps(
            clone,
            sort_keys=True
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    ########################################################
    # SAFE FEATURE BLOCK
    ########################################################

    def _safe_feature_block(self, dataset: pd.DataFrame):

        if list(dataset.columns) != list(MODEL_FEATURES):
            raise RuntimeError("Feature ordering violated.")

        block = dataset.loc[:, MODEL_FEATURES].copy()

        for col in MODEL_FEATURES:

            block[col] = pd.to_numeric(
                block[col],
                errors="coerce"
            ).astype(DTYPE)

            block[col] = block[col].replace(
                [np.inf, -np.inf],
                np.nan
            )

        if not np.isfinite(block.to_numpy()).any():
            raise RuntimeError("All features invalid.")

        return block

    ########################################################
    # CREATE BASELINE (FINAL SAFE VERSION)
    ########################################################

    def create_baseline(
        self,
        dataset: pd.DataFrame,
        dataset_hash: str,
        allow_overwrite: bool = False
    ):

        path = self._safe_baseline_path()

        if os.path.exists(path) and not allow_overwrite:
            raise RuntimeError(
                "Baseline already exists. "
                "Use allow_overwrite=True to replace it."
            )

        if len(dataset) < self.MIN_SAMPLE_BASELINE:
            raise RuntimeError(
                "Dataset too small for baseline."
            )

        logger.info("Creating drift baseline...")

        numeric = self._safe_feature_block(dataset)

        features = {}

        for col in MODEL_FEATURES:

            series = numeric[col].dropna()

            features[col] = {
                "mean": float(series.mean()),
                "std": float(max(series.std(), self.EPSILON)),
                "variance": float(max(series.var(), self.EPSILON)),
                "distribution": np.percentile(
                    series,
                    np.linspace(0, 100, 21)
                ).tolist()
            }

        payload = {

            "features": features,

            "meta": {
                "created_at": datetime.utcnow().isoformat(),
                "baseline_version": self.BASELINE_VERSION,
                "schema_signature": get_schema_signature(),
                "dataset_hash": dataset_hash,
                "rows": int(len(numeric))
            }
        }

        payload["integrity_hash"] = self._baseline_hash(payload)

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Drift baseline created successfully.")

    ########################################################
    # LOAD VERIFIED BASELINE
    ########################################################

    def _load_verified_baseline(self):

        path = self._safe_baseline_path()

        if not os.path.exists(path):
            raise RuntimeError("Baseline missing.")

        with open(path) as f:
            baseline = json.load(f)

        if baseline["integrity_hash"] != self._baseline_hash(baseline):
            raise RuntimeError("Baseline integrity failure.")

        meta = baseline["meta"]

        if meta["baseline_version"] != self.BASELINE_VERSION:
            raise RuntimeError("Baseline version mismatch.")

        if meta["schema_signature"] != get_schema_signature():
            raise RuntimeError("Baseline schema mismatch.")

        return baseline

    ########################################################
    # ACTIVE MODEL LINEAGE CHECK
    ########################################################

    def _validate_against_active_model(self, baseline):

        loader = ModelLoader()
        container = loader._xgb_container

        if container is None:
            raise RuntimeError("Active model not loaded.")

        if container.dataset_hash != baseline["meta"]["dataset_hash"]:
            raise RuntimeError(
                "Baseline dataset mismatch with active model."
            )

    ########################################################
    # DETECT DRIFT
    ########################################################

    def detect(self, dataset: pd.DataFrame):

        try:

            if dataset.empty:
                raise RuntimeError("Empty dataset supplied.")

            baseline = self._load_verified_baseline()
            self._validate_against_active_model(baseline)

            numeric = self._safe_feature_block(
                dataset.tail(self.MAX_INFERENCE_ROWS)
            )

            drift_detected = False
            report = {}

            total_features = len(baseline["features"])
            active_features = 0

            for col, stats in baseline["features"].items():

                current = numeric[col].dropna()

                if len(current) < self.MIN_SAMPLE_INFERENCE:
                    continue

                active_features += 1

                baseline_std = max(stats["std"], self.EPSILON)
                baseline_var = max(stats["variance"], self.EPSILON)
                current_var = max(current.var(), self.EPSILON)

                z_score = abs(current.mean() - stats["mean"]) / baseline_std
                variance_ratio = current_var / baseline_var
                psi = self._psi(stats["distribution"], current.values)

                drift = any([
                    z_score > self.z_threshold,
                    variance_ratio > self.VARIANCE_RATIO_UPPER,
                    variance_ratio < self.VARIANCE_RATIO_LOWER,
                    psi > self.PSI_ALERT
                ])

                if drift:
                    drift_detected = True

                report[col] = {
                    "z_score": float(z_score),
                    "variance_ratio": float(variance_ratio),
                    "psi": psi,
                    "drift": drift
                }

            coverage = active_features / max(total_features, 1)

            if coverage < self.MIN_ACTIVE_FEATURE_RATIO:
                raise RuntimeError(
                    "Feature coverage collapsed — inference unsafe."
                )

            DRIFT_DETECTED.set(1 if drift_detected else 0)

            if drift_detected and self.hard_fail:
                raise RuntimeError("Drift detected — blocking inference.")

            return {
                "drift_detected": drift_detected,
                "coverage": coverage,
                "details": report
            }

        except Exception as exc:

            logger.exception("Drift enforcement triggered: %s", exc)

            DRIFT_DETECTED.set(1)

            if self.hard_fail:
                raise

            return {
                "drift_detected": True,
                "reason": "detector_failure"
            }
