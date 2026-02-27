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

from core.artifacts.metadata_manager import MetadataManager
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

    BASELINE_FILENAME = "baseline.json"
    BASELINE_VERSION = "21.1"
    DEFAULT_BASELINE_DIR = os.path.realpath("artifacts/drift")

    MIN_SAMPLE_BASELINE = 150
    MIN_SAMPLE_INFERENCE = 20

    VARIANCE_RATIO_UPPER = 3.5
    VARIANCE_RATIO_LOWER = 0.25

    PSI_ALERT = 0.30
    MIN_BIN_PCT = 0.01

    MAX_INFERENCE_ROWS = 600

    EPSILON = 1e-8
    MAX_SEVERITY_CAP = 10

    SOFT_SEVERITY_THRESHOLD = 3
    HARD_SEVERITY_THRESHOLD = 7

    AUTO_REGENERATE = True

    def __init__(self, z_threshold: float = 3.5, baseline_dir: str = "artifacts/drift"):

        self.z_threshold = z_threshold
        self.baseline_dir = os.path.realpath(baseline_dir)
        os.makedirs(self.baseline_dir, exist_ok=True)

        self.BASELINE_PATH = os.path.join(
            self.baseline_dir,
            self.BASELINE_FILENAME
        )

        self.hard_fail = os.getenv(
            "DRIFT_HARD_FAIL",
            "false"
        ).lower() == "true"

        self._model_loader = ModelLoader()

    ########################################################
    # BASELINE CREATION
    ########################################################

    def create_baseline(
        self,
        dataset: pd.DataFrame,
        dataset_hash: str,
        training_code_hash: str,
        feature_checksum: str,
        model_version: str,
        allow_overwrite: bool = False
    ):

        if os.path.exists(self.BASELINE_PATH) and not allow_overwrite:
            raise RuntimeError(
                "Baseline already exists. Use allow_overwrite=True to replace."
            )

        logger.info("Creating governed drift baseline.")

        numeric = self._safe_feature_block(dataset)
        baseline_features = {}

        for col in MODEL_FEATURES:

            series = numeric[col].dropna()

            if len(series) < self.MIN_SAMPLE_BASELINE:
                continue

            mean = float(series.mean())
            std = float(max(series.std(), self.EPSILON))
            variance = float(max(series.var(), self.EPSILON))

            quantiles = np.linspace(0, 1, 11)
            bin_edges = np.unique(np.quantile(series, quantiles))

            if len(bin_edges) < 2:
                bin_edges = np.array([series.min(), series.max() + self.EPSILON])

            expected_counts = np.histogram(series, bins=bin_edges)[0]

            baseline_features[col] = {
                "mean": mean,
                "std": std,
                "variance": variance,
                "bin_edges": bin_edges.tolist(),
                "expected_counts": expected_counts.tolist()
            }

        if len(baseline_features) == 0:
            raise RuntimeError("Baseline feature set empty — insufficient training data.")

        payload = {
            "meta": {
                "baseline_version": self.BASELINE_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "schema_signature": get_schema_signature(),
                "model_version": model_version,
                "dataset_hash": dataset_hash,
                "training_code_hash": training_code_hash,
                "feature_checksum": feature_checksum,
                "feature_count": len(baseline_features)
            },
            "features": baseline_features
        }

        self._atomic_write(payload)

    ########################################################
    # SAFE FEATURE BLOCK
    ########################################################

    def _safe_feature_block(self, dataset: pd.DataFrame):

        missing = set(MODEL_FEATURES) - set(dataset.columns)
        if missing:
            raise RuntimeError(f"Missing features: {missing}")

        block = dataset.loc[:, MODEL_FEATURES].copy()

        for col in MODEL_FEATURES:
            block[col] = pd.to_numeric(block[col], errors="coerce").astype(DTYPE)
            block[col] = block[col].replace([np.inf, -np.inf], np.nan)

        return block

    ########################################################
    # HASH
    ########################################################

    @staticmethod
    def _baseline_hash(payload: dict) -> str:

        clone = dict(payload)
        clone.pop("integrity_hash", None)

        canonical = json.dumps(clone, sort_keys=True).encode()
        return hashlib.sha256(canonical).hexdigest()

    ########################################################
    # ATOMIC WRITE
    ########################################################

    def _atomic_write(self, payload: dict):

        payload["integrity_hash"] = self._baseline_hash(payload)

        tmp_path = self.BASELINE_PATH + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        os.replace(tmp_path, self.BASELINE_PATH)

    ########################################################
    # LOAD VERIFIED BASELINE
    ########################################################

    def _load_verified_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise RuntimeError("Baseline missing.")

        with open(self.BASELINE_PATH, encoding="utf-8") as f:
            baseline = json.load(f)

        if baseline["integrity_hash"] != self._baseline_hash(baseline):
            raise RuntimeError("Baseline integrity failure.")

        meta = baseline["meta"]

        if meta["baseline_version"] != self.BASELINE_VERSION:
            raise RuntimeError("Baseline version mismatch.")

        if meta["schema_signature"] != get_schema_signature():
            raise RuntimeError("Baseline schema mismatch.")

        current_checksum = MetadataManager.fingerprint_features(
            tuple(MODEL_FEATURES)
        )

        if meta.get("feature_checksum") != current_checksum:
            raise RuntimeError("Feature checksum mismatch.")

        return baseline

    ########################################################
    # PSI
    ########################################################

    def _psi(self, bin_edges, expected_counts, actual):

        actual = np.asarray(actual, dtype=np.float64)

        if len(actual) < 5:
            return 0.0

        actual_counts = np.histogram(actual, bins=bin_edges)[0]

        expected_perc = expected_counts / max(expected_counts.sum(), self.EPSILON)
        actual_perc = actual_counts / max(actual_counts.sum(), self.EPSILON)

        expected_perc = np.clip(expected_perc, self.MIN_BIN_PCT, None)
        actual_perc = np.clip(actual_perc, self.MIN_BIN_PCT, None)

        psi = np.sum(
            (actual_perc - expected_perc) *
            np.log((actual_perc + self.EPSILON) /
                   (expected_perc + self.EPSILON))
        )

        return float(psi)

    ########################################################
    # DETECT
    ########################################################

    def detect(self, dataset: pd.DataFrame):

        try:

            baseline = self._load_verified_baseline()

            numeric = self._safe_feature_block(
                dataset.tail(self.MAX_INFERENCE_ROWS)
            )

            drift_count = 0
            severity_accumulator = 0
            report = {}

            total_features = len(baseline["features"])
            evaluated_features = 0

            for col, stats in baseline["features"].items():

                current = numeric[col].dropna()

                if len(current) < self.MIN_SAMPLE_INFERENCE:
                    continue

                evaluated_features += 1

                baseline_std = max(stats["std"], self.EPSILON)
                baseline_var = max(stats["variance"], self.EPSILON)
                current_var = max(current.var(), self.EPSILON)

                z_score = abs(current.mean() - stats["mean"]) / baseline_std
                variance_ratio = current_var / baseline_var

                psi = self._psi(
                    np.array(stats["bin_edges"]),
                    np.array(stats["expected_counts"]),
                    current.values
                )

                drift = any([
                    z_score > self.z_threshold,
                    variance_ratio > self.VARIANCE_RATIO_UPPER,
                    variance_ratio < self.VARIANCE_RATIO_LOWER,
                    psi > self.PSI_ALERT
                ])

                if drift:
                    drift_count += 1
                    severity_accumulator += (
                        min(z_score / self.z_threshold, 3) +
                        min(abs(np.log(variance_ratio + self.EPSILON)), 3) +
                        min(psi / self.PSI_ALERT, 3)
                    )

                report[col] = {
                    "z_score": float(z_score),
                    "variance_ratio": float(variance_ratio),
                    "psi": float(psi),
                    "drift": drift
                }

            severity_score = min(int(severity_accumulator), self.MAX_SEVERITY_CAP)

            drift_state = (
                "hard" if severity_score >= self.HARD_SEVERITY_THRESHOLD
                else "soft" if severity_score >= self.SOFT_SEVERITY_THRESHOLD
                else "none"
            )

            drift_detected = drift_count > 0
            coverage = float(evaluated_features / max(total_features, 1))

            DRIFT_DETECTED.set(1 if drift_detected else 0)

            return {
                "drift_detected": drift_detected,
                "severity_score": severity_score,
                "drift_confidence": float(min(drift_count / max(total_features, 1), 1.0)),
                "coverage": coverage,
                "details": report,
                "drift_state": drift_state
            }

        except Exception as exc:

            logger.warning("Drift detector failure: %s", exc)

            if self.hard_fail:
                raise

            DRIFT_DETECTED.set(1)

            return {
                "drift_detected": True,
                "severity_score": 5,
                "drift_confidence": 0.5,
                "coverage": 0.0,
                "details": {},
                "drift_state": "detector_failure",
                "reason": "detector_failure"
            }