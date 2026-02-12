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
    BASELINE_VERSION = "6.3"

    MIN_SAMPLE_BASELINE = 50
    MIN_SAMPLE_INFERENCE = 20

    VARIANCE_RATIO_UPPER = 2.5
    VARIANCE_RATIO_LOWER = 0.4

    MAX_INFERENCE_ROWS = 500

    EPSILON = 1e-6

    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        os.makedirs("artifacts/drift", exist_ok=True)

    ########################################################

    @staticmethod
    def _atomic_write(path: str, payload: dict):

        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

        if os.name != "nt":
            fd = os.open(directory, os.O_DIRECTORY)
            os.fsync(fd)
            os.close(fd)

    ########################################################
    # STRICT FEATURE BLOCK
    ########################################################

    def _safe_feature_block(self, dataset: pd.DataFrame):

        if dataset.columns.duplicated().any():
            raise RuntimeError("Duplicate columns detected in dataset.")

        incoming = set(dataset.columns)
        expected = set(MODEL_FEATURES)

        missing = expected - incoming
        unknown = incoming - expected

        if missing:
            raise RuntimeError(
                f"Drift detector schema violation. Missing={missing}"
            )

        if unknown:
            raise RuntimeError(
                f"Unknown features detected in inference data: {unknown}"
            )

        block = dataset.loc[:, MODEL_FEATURES].copy(deep=True)

        for col in MODEL_FEATURES:

            block[col] = pd.to_numeric(
                block[col],
                errors="coerce"
            ).astype(DTYPE)

            finite = np.isfinite(block[col])

            if not finite.any():
                raise RuntimeError(
                    f"No finite values in feature '{col}'."
                )

        if list(block.columns) != list(MODEL_FEATURES):
            raise RuntimeError("Feature ordering enforcement failed.")

        return block

    ########################################################
    # STRUCTURAL HASH
    ########################################################

    @staticmethod
    def _dataset_sha256(df: pd.DataFrame) -> str:

        ordered = df.reindex(columns=sorted(df.columns)).copy()
        ordered = ordered.round(8)

        hasher = hashlib.sha256()

        hasher.update(",".join(ordered.columns).encode())

        hashed = pd.util.hash_pandas_object(
            ordered,
            index=False
        ).values

        hasher.update(hashed.tobytes())

        return hasher.hexdigest()

    ########################################################

    def _validate_baseline_structure(self, baseline: dict):

        if "meta" not in baseline or "features" not in baseline:
            raise RuntimeError("Baseline structure corrupted.")

        if baseline["meta"].get("baseline_version") != self.BASELINE_VERSION:
            raise RuntimeError("Baseline version mismatch.")

        for feature in MODEL_FEATURES:
            if feature not in baseline["features"]:
                raise RuntimeError(
                    f"Baseline missing feature stats: {feature}"
                )

    ########################################################

    def create_baseline(
        self,
        dataset: pd.DataFrame,
        dataset_hash: str | None = None,
        allow_overwrite: bool = False
    ):

        if dataset.empty:
            raise RuntimeError("Cannot create drift baseline from empty dataset.")

        numeric = self._safe_feature_block(dataset)

        if dataset_hash is None:
            dataset_hash = self._dataset_sha256(numeric)

        if os.path.exists(self.BASELINE_PATH):

            if not allow_overwrite:

                try:
                    with open(self.BASELINE_PATH) as f:
                        existing = json.load(f)

                    self._validate_baseline_structure(existing)

                    if existing.get("meta", {}).get("dataset_hash") == dataset_hash:
                        logger.info("Baseline already matches dataset.")
                        return

                except Exception:
                    pass

                raise RuntimeError(
                    "Baseline exists with different lineage. "
                    "Use allow_overwrite=True after retraining."
                )

        baseline: Dict[str, Any] = {
            "meta": {
                "baseline_version": self.BASELINE_VERSION,
                "schema_signature": get_schema_signature(),
                "schema_version": SCHEMA_VERSION,
                "dataset_hash": dataset_hash,
                "feature_count": len(MODEL_FEATURES),
                "created_at": pd.Timestamp.utcnow().isoformat()
            },
            "features": {}
        }

        for col in MODEL_FEATURES:

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

        self._atomic_write(self.BASELINE_PATH, baseline)

        logger.info("Drift baseline created.")

    ########################################################

    def _load_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise RuntimeError("Drift baseline missing.")

        with open(self.BASELINE_PATH, "r") as f:
            baseline = json.load(f)

        self._validate_baseline_structure(baseline)

        meta = baseline.get("meta", {})

        if meta.get("schema_signature") != get_schema_signature():
            raise RuntimeError(
                "Baseline schema mismatch. Retraining required."
            )

        if meta.get("feature_count") != len(MODEL_FEATURES):
            raise RuntimeError(
                "Baseline feature count mismatch."
            )

        return baseline

    ########################################################

    def detect(self, dataset: pd.DataFrame):

        try:

            if dataset.empty:
                logger.warning("Drift skipped — empty dataset.")
                return {"drift_detected": False, "details": {}}

            baseline = self._load_baseline()

            dataset = dataset.tail(self.MAX_INFERENCE_ROWS)

            numeric = self._safe_feature_block(dataset)

            drift_detected = False
            drift_report = {}

            for col in MODEL_FEATURES:

                stats = baseline["features"][col]
                current = numeric[col].dropna()

                if len(current) < self.MIN_SAMPLE_INFERENCE:
                    continue

                var_now = current.var()

                if var_now < self.EPSILON:
                    drift_detected = True
                    drift_report[col] = {"variance_collapse": True}
                    continue

                mean_now = current.mean()

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

        except Exception as exc:

            logger.exception(
                f"Drift detector failure — forcing alert. Root cause: {exc}"
            )

            DRIFT_DETECTED.set(1)

            return {
                "drift_detected": True,
                "reason": "detector_failure"
            }
