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
    BASELINE_VERSION = "16.0"

    MIN_SAMPLE_BASELINE = 150
    MIN_SAMPLE_INFERENCE = 40

    VARIANCE_RATIO_UPPER = 3.5
    VARIANCE_RATIO_LOWER = 0.25

    PSI_ALERT = 0.30
    MIN_BIN_PCT = 0.01

    MAX_INFERENCE_ROWS = 600

    EPSILON = 1e-8
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
    # CREATE BASELINE (DECOUPLED FROM MODEL LOADER)
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

        if dataset is None or dataset.empty:
            raise RuntimeError("Cannot build baseline from empty dataset.")

        path = self._safe_baseline_path()

        if os.path.exists(path) and not allow_overwrite:
            raise RuntimeError("Baseline already exists.")

        numeric = self._safe_feature_block(dataset)

        if len(numeric) < self.MIN_SAMPLE_BASELINE:
            raise RuntimeError("Insufficient rows for baseline creation.")

        features = {}

        for col in MODEL_FEATURES:

            series = numeric[col].dropna()

            if len(series) < self.MIN_SAMPLE_BASELINE:
                raise RuntimeError(f"Baseline insufficient data for {col}")

            series = (
                series
                .astype(np.float64)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .round(10)
            )

            mean = float(series.mean())
            std = float(series.std())
            variance = float(series.var())

            if not np.isfinite([mean, std, variance]).all():
                raise RuntimeError(f"Non-finite baseline stats for {col}")

            counts, bin_edges = np.histogram(series, bins=20)

            if len(bin_edges) < 2:
                raise RuntimeError("Invalid histogram construction.")

            features[col] = {
                "mean": mean,
                "std": std,
                "variance": variance,
                "bin_edges": bin_edges.tolist(),
                "expected_counts": counts.tolist()
            }

        payload = {
            "meta": {
                "baseline_version": self.BASELINE_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "schema_signature": get_schema_signature(),
                "dataset_hash": dataset_hash,
                "training_code_hash": training_code_hash,
                "model_feature_checksum": feature_checksum,
                "model_version": model_version
            },
            "features": features
        }

        payload["integrity_hash"] = self._baseline_hash(payload)

        self._atomic_write(payload, path)

        logger.info(
            "Drift baseline created for model_version=%s",
            model_version
        )

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
    # ATOMIC WRITE
    ########################################################

    def _atomic_write(self, payload, path):

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=os.path.dirname(path),
            suffix=".tmp"
        ) as tmp:

            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name

        os.replace(temp_name, path)

    ########################################################
    # PSI
    ########################################################

    def _psi(self, bin_edges, expected_counts, actual):

        actual = np.asarray(actual, dtype=np.float64)

        actual_counts = np.histogram(actual, bins=bin_edges)[0]

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
            np.log((actual_perc + self.EPSILON) /
                   (expected_perc + self.EPSILON))
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
    # LOAD VERIFIED BASELINE
    ########################################################

    def _load_verified_baseline(self):

        path = self._safe_baseline_path()

        if not os.path.exists(path):
            raise RuntimeError("Baseline missing.")

        with open(path, encoding="utf-8") as f:
            baseline = json.load(f)

        if baseline["integrity_hash"] != self._baseline_hash(baseline):
            raise RuntimeError("Baseline integrity failure.")

        meta = baseline["meta"]

        if meta["baseline_version"] != self.BASELINE_VERSION:
            raise RuntimeError("Baseline version mismatch.")

        if meta["schema_signature"] != get_schema_signature():
            raise RuntimeError("Baseline schema mismatch.")

        loader = self._model_loader

        if meta.get("model_version") != loader.xgb_version:
            raise RuntimeError(
                f"Baseline tied to model_version={meta.get('model_version')} "
                f"but current production version={loader.xgb_version}"
            )

        if meta.get("model_feature_checksum") != loader.feature_checksum:
            raise RuntimeError("Model feature checksum mismatch.")

        if meta["dataset_hash"] != loader.dataset_hash:
            logger.warning("Dataset hash drift detected.")

        if meta["training_code_hash"] != loader.training_code_hash:
            logger.warning("Training code drift detected.")

        return baseline

    ########################################################
    # DETECT DRIFT (UNCHANGED)
    ########################################################

    def detect(self, dataset: pd.DataFrame):

        try:

            if dataset.empty:
                raise RuntimeError("Empty dataset supplied.")

            baseline = self._load_verified_baseline()

            numeric = self._safe_feature_block(
                dataset.tail(self.MAX_INFERENCE_ROWS)
            )

            drift_detected = False
            severity_score = 0
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
                    severity_score += 1

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
                logger.critical(
                    "FEATURE DRIFT DETECTED | severity=%s",
                    severity_score
                )

            return {
                "drift_detected": drift_detected,
                "severity_score": severity_score,
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