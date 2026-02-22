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
    BASELINE_VERSION = "17.1"   # 🔥 bumped due to contract fix
    DEFAULT_BASELINE_DIR = os.path.realpath("artifacts/drift")

    MIN_SAMPLE_BASELINE = 150
    MIN_SAMPLE_INFERENCE = 20

    VARIANCE_RATIO_UPPER = 3.5
    VARIANCE_RATIO_LOWER = 0.25

    PSI_ALERT = 0.30
    MIN_BIN_PCT = 0.01

    MAX_INFERENCE_ROWS = 600

    EPSILON = 1e-8
    MIN_ACTIVE_FEATURE_RATIO = 0.40
    MAX_SEVERITY_CAP = 10

    ########################################################

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

        self._enforce_model_contract = (
            os.path.realpath(self.baseline_dir) ==
            self.DEFAULT_BASELINE_DIR
        )

    ########################################################
    # BASELINE CREATION
    ########################################################

    def create_baseline(
        self,
        dataset: pd.DataFrame,
        model_version: str,
        dataset_hash: str | None = None,
        training_code_hash: str | None = None,
        feature_checksum: str | None = None,
        allow_overwrite: bool = False,
    ):

        if os.path.exists(self.BASELINE_PATH) and not allow_overwrite:
            raise RuntimeError(
                "Baseline already exists. Use allow_overwrite=True to replace."
            )

        if dataset is None or dataset.empty:
            raise RuntimeError("Cannot create baseline from empty dataset.")

        if len(dataset) < self.MIN_SAMPLE_BASELINE:
            raise RuntimeError(
                f"Insufficient rows for baseline creation. "
                f"Minimum required: {self.MIN_SAMPLE_BASELINE}"
            )

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

        if not baseline_features:
            raise RuntimeError("Baseline generation failed — no valid features.")

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

        payload["integrity_hash"] = self._baseline_hash(payload)

        self._atomic_write(payload)

        logger.info(
            "Drift baseline created successfully | features=%s",
            len(baseline_features)
        )

    ########################################################
    # ATOMIC WRITE
    ########################################################

    def _atomic_write(self, payload: dict):

        tmp_fd, tmp_path = tempfile.mkstemp()

        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self.BASELINE_PATH)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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

        return block

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
    # LOAD BASELINE
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

        if "feature_checksum" in meta and meta["feature_checksum"]:
            current_checksum = MetadataManager.fingerprint_features(
                tuple(MODEL_FEATURES)
            )
            if meta["feature_checksum"] != current_checksum:
                raise RuntimeError("Baseline feature checksum mismatch.")

        return baseline

    ########################################################
    # DETECT
    ########################################################

    def detect(self, dataset: pd.DataFrame):

        try:

            if dataset.empty:
                raise RuntimeError("Empty dataset supplied.")

            baseline = self._load_verified_baseline()

            numeric = self._safe_feature_block(
                dataset.tail(self.MAX_INFERENCE_ROWS)
            )

            drift_count = 0
            report = {}

            total_features = len(baseline["features"])
            active_features = 0

            sample_size = len(numeric)

            for col, stats in baseline["features"].items():

                current = numeric[col].dropna()

                # 🔧 Adaptive minimum sample logic
                if len(current) < min(self.MIN_SAMPLE_INFERENCE, sample_size):
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
                    drift_count += 1

                report[col] = {
                    "z_score": float(z_score),
                    "variance_ratio": float(variance_ratio),
                    "psi": psi,
                    "drift": drift
                }

            coverage = active_features / max(total_features, 1)

            # 🔧 Only fail if truly catastrophic
            if sample_size >= self.MIN_SAMPLE_INFERENCE:
                if coverage < self.MIN_ACTIVE_FEATURE_RATIO:
                    raise RuntimeError(
                        "Feature coverage collapsed — inference unsafe."
                    )

            # 🔧 Normalize severity
            severity_ratio = drift_count / max(total_features, 1)
            severity_score = min(
                int(severity_ratio * 10),
                self.MAX_SEVERITY_CAP
            )

            drift_detected = drift_count > 0

            DRIFT_DETECTED.set(1 if drift_detected else 0)

            if drift_detected:
                logger.critical(
                    "FEATURE DRIFT DETECTED | severity=%s | coverage=%.2f",
                    severity_score,
                    coverage
                )

            return {
                "drift_detected": drift_detected,
                "severity_score": int(severity_score),
                "coverage": float(coverage),
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