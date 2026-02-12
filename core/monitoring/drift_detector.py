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
    BASELINE_VERSION = "7.0"   # 🔥 bump version

    MIN_SAMPLE_BASELINE = 50
    MIN_SAMPLE_INFERENCE = 20

    VARIANCE_RATIO_UPPER = 3.0
    VARIANCE_RATIO_LOWER = 0.30

    MAX_INFERENCE_ROWS = 500

    EPSILON = 1e-6
    SOFT_VARIANCE_FLOOR = 1e-5   # 🔥 institutional tolerance

    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        os.makedirs("artifacts/drift", exist_ok=True)

    ########################################################
    # ATOMIC WRITE
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
    # SAFE FEATURE BLOCK
    ########################################################

    def _safe_feature_block(self, dataset: pd.DataFrame):

        if dataset.columns.duplicated().any():
            raise RuntimeError("Duplicate columns detected.")

        incoming = set(dataset.columns)
        expected = set(MODEL_FEATURES)

        if expected - incoming:
            raise RuntimeError("Drift schema violation.")

        block = dataset.loc[:, MODEL_FEATURES].copy()

        for col in MODEL_FEATURES:

            block[col] = pd.to_numeric(
                block[col],
                errors="coerce"
            )

            # ✅ pandas 3 safe
            block[col] = block[col].replace(
                [np.inf, -np.inf],
                np.nan
            )

            block[col] = block[col].astype(DTYPE)

            if not np.isfinite(block[col]).any():
                raise RuntimeError(
                    f"No finite values in feature '{col}'."
                )

        return block

    ########################################################
    # HASH
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

    def create_baseline(
        self,
        dataset: pd.DataFrame,
        dataset_hash: str | None = None,
        allow_overwrite: bool = False
    ):

        if dataset.empty:
            raise RuntimeError("Cannot baseline empty dataset.")

        numeric = self._safe_feature_block(dataset)

        if dataset_hash is None:
            dataset_hash = self._dataset_sha256(numeric)

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

        skipped_features = []

        for col in MODEL_FEATURES:

            series = numeric[col].dropna()

            if len(series) < self.MIN_SAMPLE_BASELINE:
                logger.warning(
                    "Skipping baseline feature '%s' — insufficient samples.",
                    col
                )
                skipped_features.append(col)
                continue

            std = series.std()

            ####################################################
            # 🔥 SOFT VARIANCE GUARD
            ####################################################

            if std < self.SOFT_VARIANCE_FLOOR:

                logger.warning(
                    "Skipping baseline feature '%s' — low variance (%.8f).",
                    col,
                    std
                )

                skipped_features.append(col)
                continue

            baseline["features"][col] = {
                "mean": float(series.mean()),
                "std": float(std),
                "variance": float(series.var()),
                "min": float(series.min()),
                "max": float(series.max()),
                "count": int(len(series))
            }

        if len(baseline["features"]) < len(MODEL_FEATURES) * 0.6:
            raise RuntimeError(
                "Too many unstable features — training unsafe."
            )

        baseline["meta"]["skipped_features"] = skipped_features

        self._atomic_write(self.BASELINE_PATH, baseline)

        logger.info(
            "Drift baseline created. Active=%d Skipped=%d",
            len(baseline["features"]),
            len(skipped_features)
        )

    ########################################################

    def detect(self, dataset: pd.DataFrame):

        try:

            if dataset.empty:
                return {"drift_detected": False, "details": {}}

            if not os.path.exists(self.BASELINE_PATH):
                raise RuntimeError("Baseline missing.")

            with open(self.BASELINE_PATH) as f:
                baseline = json.load(f)

            numeric = self._safe_feature_block(
                dataset.tail(self.MAX_INFERENCE_ROWS)
            )

            drift_detected = False
            report = {}

            for col, stats in baseline["features"].items():

                current = numeric[col].dropna()

                if len(current) < self.MIN_SAMPLE_INFERENCE:
                    continue

                var_now = max(current.var(), self.EPSILON)

                z_score = abs(
                    current.mean() - stats["mean"]
                ) / max(stats["std"], self.EPSILON)

                variance_ratio = var_now / max(
                    stats["variance"],
                    self.EPSILON
                )

                drift = (
                    z_score > self.z_threshold
                    or variance_ratio > self.VARIANCE_RATIO_UPPER
                    or variance_ratio < self.VARIANCE_RATIO_LOWER
                )

                if drift:
                    drift_detected = True

                report[col] = {
                    "z_score": float(z_score),
                    "variance_ratio": float(variance_ratio),
                    "drift": drift
                }

            DRIFT_DETECTED.set(1 if drift_detected else 0)

            return {
                "drift_detected": drift_detected,
                "details": report
            }

        except Exception as exc:

            logger.exception(
                "Drift detector failure — forcing alert: %s",
                exc
            )

            DRIFT_DETECTED.set(1)

            return {
                "drift_detected": True,
                "reason": "detector_failure"
            }
