import numpy as np
import pandas as pd
import os
import json
import logging
import hashlib
import tempfile
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
    BASELINE_VERSION = "14.0"

    MIN_SAMPLE_BASELINE = 150
    MIN_SAMPLE_INFERENCE = 40

    VARIANCE_RATIO_UPPER = 3.5
    VARIANCE_RATIO_LOWER = 0.25

    PSI_ALERT = 0.30
    MIN_BIN_PCT = 0.01

    MAX_INFERENCE_ROWS = 600

    EPSILON = 1e-6
    MIN_ACTIVE_FEATURE_RATIO = 0.60

    ########################################################

    def __init__(self, z_threshold: float = 3.5):

        self.z_threshold = z_threshold
        os.makedirs("artifacts/drift", exist_ok=True)

        self.hard_fail = os.getenv(
            "DRIFT_HARD_FAIL",
            "false"
        ).lower() == "true"

        self._model_loader = ModelLoader()

    ########################################################
    # SAFE PATH
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

    def _psi(self, bin_edges, expected_counts, actual):

        actual_counts = np.histogram(
            actual,
            bins=bin_edges
        )[0]

        expected_perc = expected_counts / max(
            expected_counts.sum(), self.EPSILON
        )

        actual_perc = actual_counts / max(
            actual_counts.sum(), self.EPSILON
        )

        expected_perc = np.clip(expected_perc, self.MIN_BIN_PCT, None)
        actual_perc = np.clip(actual_perc, self.MIN_BIN_PCT, None)

        psi = np.sum(
            (actual_perc - expected_perc) *
            np.log(actual_perc / expected_perc)
        )

        return float(psi)

    ########################################################
    # HASH
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

        missing = set(MODEL_FEATURES) - set(dataset.columns)

        if missing:
            raise RuntimeError(
                f"Missing features for drift detection: {missing}"
            )

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
    # ATOMIC WRITE
    ########################################################

    def _atomic_write(self, payload, path):

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=os.path.dirname(path),
            suffix=".tmp"
        ) as tmp:

            json.dump(payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())

            temp_name = tmp.name

        os.replace(temp_name, path)

    ########################################################
    # CREATE BASELINE
    ########################################################

    def create_baseline(
        self,
        dataset: pd.DataFrame,
        dataset_hash: str,
        training_code_hash: str,
        allow_overwrite: bool = False
    ):

        path = self._safe_baseline_path()

        if os.path.exists(path) and not allow_overwrite:
            raise RuntimeError("Baseline already exists.")

        if len(dataset) < self.MIN_SAMPLE_BASELINE:
            raise RuntimeError("Dataset too small for baseline.")

        numeric = self._safe_feature_block(dataset)

        features = {}

        for col in MODEL_FEATURES:

            series = numeric[col].dropna()

            counts, edges = np.histogram(series, bins=25)

            features[col] = {
                "mean": float(series.mean()),
                "std": float(max(series.std(), self.EPSILON)),
                "variance": float(max(series.var(), self.EPSILON)),
                "bin_edges": edges.tolist(),
                "expected_counts": counts.tolist()
            }

        payload = {
            "features": features,
            "meta": {
                "created_at": datetime.utcnow().isoformat(),
                "baseline_version": self.BASELINE_VERSION,
                "schema_signature": get_schema_signature(),
                "dataset_hash": dataset_hash,
                "training_code_hash": training_code_hash,
                "rows": int(len(numeric))
            }
        }

        payload["integrity_hash"] = self._baseline_hash(payload)

        self._atomic_write(payload, path)

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
    # VALIDATE AGAINST ACTIVE MODEL
    ########################################################

    def _validate_against_active_model(self, baseline):

        _ = self._model_loader.xgb  # ensures model loaded

        if self._model_loader.dataset_hash != baseline["meta"]["dataset_hash"]:
            raise RuntimeError(
                "Baseline dataset mismatch with active model."
            )

        if self._model_loader.training_code_hash != baseline["meta"]["training_code_hash"]:
            raise RuntimeError(
                "Training code hash mismatch — baseline stale."
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

            if drift_detected:
                logger.critical("FEATURE DRIFT DETECTED")

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