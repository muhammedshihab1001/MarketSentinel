import numpy as np
import pandas as pd
import os
import json
import logging
from typing import Dict, Any

from core.schema.feature_schema import (
    get_schema_signature,
    SCHEMA_VERSION,
    MODEL_FEATURES
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
    """
    Institutional Drift Sentinel — Hardened.

    Guarantees:
    ✔ schema-bound baseline
    ✔ lineage enforcement
    ✔ crash-safe writes
    ✔ fail-closed detection
    ✔ feature ordering enforcement
    ✔ training compatibility
    """

    BASELINE_PATH = "artifacts/drift/baseline.json"
    BASELINE_VERSION = "5.0"

    MIN_SAMPLE_BASELINE = 50
    MIN_SAMPLE_INFERENCE = 20

    VARIANCE_RATIO_UPPER = 2.5
    VARIANCE_RATIO_LOWER = 0.4

    EPSILON = 1e-6

    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        os.makedirs("artifacts/drift", exist_ok=True)

    # --------------------------------------------------
    # ATOMIC WRITE
    # --------------------------------------------------

    @staticmethod
    def _atomic_write(path: str, payload: dict):

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    # --------------------------------------------------
    # SAFE FEATURE EXTRACTION
    # --------------------------------------------------

    def _safe_feature_block(self, dataset: pd.DataFrame):

        missing = set(MODEL_FEATURES) - set(dataset.columns)

        if missing:
            raise RuntimeError(
                f"Drift detector schema violation. Missing={missing}"
            )

        block = dataset.loc[:, MODEL_FEATURES]

        if list(block.columns) != MODEL_FEATURES:
            raise RuntimeError("Feature ordering corrupted.")

        return block

    # --------------------------------------------------
    # BASELINE CREATION
    # --------------------------------------------------

    def create_baseline(
        self,
        dataset: pd.DataFrame,
        dataset_hash: str | None = None,
        allow_overwrite: bool = False
    ):
        """
        dataset_hash is optional for backward compatibility.
        """

        if dataset.empty:
            raise RuntimeError("Cannot create drift baseline from empty dataset.")

        numeric = self._safe_feature_block(dataset)

        # AUTO HASH if not provided (training compatibility)
        if dataset_hash is None:

            hashed = pd.util.hash_pandas_object(
                numeric.round(8),
                index=False
            ).values.tobytes()

            dataset_hash = str(hash(hashed))

        # overwrite policy
        if os.path.exists(self.BASELINE_PATH):

            if not allow_overwrite:

                try:
                    with open(self.BASELINE_PATH) as f:
                        existing = json.load(f)

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

        if meta.get("feature_count") != len(MODEL_FEATURES):
            raise RuntimeError(
                "Baseline feature count mismatch."
            )

        return baseline

    # --------------------------------------------------
    # DRIFT DETECTION
    # --------------------------------------------------

    def detect(self, dataset: pd.DataFrame):

        try:

            if dataset.empty:
                logger.warning("Drift skipped — empty dataset.")
                return {"drift_detected": False, "details": {}}

            baseline = self._load_baseline()
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

        except Exception:

            logger.exception(
                "Drift detector failure — forcing alert."
            )

            DRIFT_DETECTED.set(1)

            return {
                "drift_detected": True,
                "reason": "detector_failure"
            }
