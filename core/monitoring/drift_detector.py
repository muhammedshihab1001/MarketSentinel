import numpy as np
import pandas as pd
import os
import json
import logging
import hashlib
from typing import Dict, Any

from core.schema.feature_schema import (
    get_schema_signature,
    SCHEMA_VERSION,
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
    BASELINE_VERSION = "9.0"

    MIN_SAMPLE_BASELINE = 50
    MIN_SAMPLE_INFERENCE = 20

    VARIANCE_RATIO_UPPER = 3.0
    VARIANCE_RATIO_LOWER = 0.30

    MAX_INFERENCE_ROWS = 500

    EPSILON = 1e-6
    SOFT_VARIANCE_FLOOR = 1e-5

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
    # ATOMIC WRITE
    ########################################################

    @staticmethod
    def _atomic_write(path: str, payload: dict):

        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)

        payload["integrity_hash"] = DriftDetector._baseline_hash(payload)

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

        if os.name != "nt":
            fd = os.open(directory, os.O_DIRECTORY)
            os.fsync(fd)
            os.close(fd)

    ########################################################
    # VERIFY BASELINE
    ########################################################

    def _load_verified_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise RuntimeError("Baseline missing.")

        with open(self.BASELINE_PATH) as f:
            baseline = json.load(f)

        expected = baseline.get("integrity_hash")

        actual = self._baseline_hash(baseline)

        if expected != actual:
            raise RuntimeError("Baseline integrity failure.")

        if baseline["meta"]["baseline_version"] != self.BASELINE_VERSION:
            raise RuntimeError("Baseline version incompatible.")

        if baseline["meta"]["schema_signature"] != get_schema_signature():
            raise RuntimeError("Baseline schema mismatch.")

        return baseline

    ########################################################
    # ACTIVE MODEL LINEAGE CHECK
    ########################################################

    def _validate_against_active_model(self, baseline):

        try:
            loader = ModelLoader()
            container = loader._xgb_container

            if container is None:
                raise RuntimeError("Model not loaded.")

            active_hash = container.dataset_hash
            baseline_hash = baseline["meta"]["dataset_hash"]

            if active_hash != baseline_hash:
                raise RuntimeError(
                    "Baseline dataset does not match active model."
                )

        except Exception as e:
            raise RuntimeError(
                f"Active model lineage validation failed: {e}"
            )

    ########################################################
    # SAFE FEATURE BLOCK
    ########################################################

    def _safe_feature_block(self, dataset: pd.DataFrame):

        incoming = set(dataset.columns)
        expected = set(MODEL_FEATURES)

        missing = expected - incoming
        unknown = incoming - expected

        if missing:
            raise RuntimeError(f"Missing features: {missing}")

        if unknown:
            raise RuntimeError(f"Unknown features detected: {unknown}")

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
    # DETECT
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

            active_features = 0

            for col, stats in baseline["features"].items():

                current = numeric[col].dropna()

                if len(current) < self.MIN_SAMPLE_INFERENCE:
                    continue

                active_features += 1

                var_now = max(current.var(), self.EPSILON)

                mean_now = current.mean()

                z_score = abs(mean_now - stats["mean"]) / max(
                    stats["std"], self.EPSILON
                )

                variance_ratio = var_now / max(
                    stats["variance"], self.EPSILON
                )

                drift = any([
                    z_score > self.z_threshold,
                    variance_ratio > self.VARIANCE_RATIO_UPPER,
                    variance_ratio < self.VARIANCE_RATIO_LOWER,
                    current.min() < stats["min"],
                    current.max() > stats["max"]
                ])

                if drift:
                    drift_detected = True

                report[col] = {
                    "z_score": float(z_score),
                    "variance_ratio": float(variance_ratio),
                    "drift": drift
                }

            coverage = active_features / len(baseline["features"])

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
