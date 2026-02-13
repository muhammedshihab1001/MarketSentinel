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

logger = logging.getLogger("marketsentinel.drift")


class DriftDetector:

    BASELINE_PATH = "artifacts/drift/baseline.json"
    BASELINE_VERSION = "11.0"

    MIN_SAMPLE_BASELINE = 200
    MIN_SAMPLE_INFERENCE = 30

    VARIANCE_RATIO_UPPER = 3.0
    VARIANCE_RATIO_LOWER = 0.30

    PSI_ALERT = 0.25
    MAX_INFERENCE_ROWS = 500

    EPSILON = 1e-6

    ########################################################

    def __init__(self, z_threshold: float = 3.0):

        self.z_threshold = z_threshold
        os.makedirs("artifacts/drift", exist_ok=True)

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

        block = block.dropna()

        if len(block) < self.MIN_SAMPLE_BASELINE:
            raise RuntimeError(
                "Not enough samples to create drift baseline."
            )

        return block

    ########################################################
    # CREATE BASELINE  ⭐⭐⭐⭐⭐
    ########################################################

    def create_baseline(self, dataset: pd.DataFrame, dataset_hash: str):

        logger.info("Creating drift baseline...")

        numeric = self._safe_feature_block(dataset)

        features = {}

        for col in MODEL_FEATURES:

            series = numeric[col]

            features[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "variance": float(series.var()),
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

        path = os.path.realpath(self.BASELINE_PATH)

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Drift baseline created successfully.")

    ########################################################
    # LOAD BASELINE
    ########################################################

    def _load_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise RuntimeError("Baseline missing.")

        with open(self.BASELINE_PATH) as f:
            baseline = json.load(f)

        if baseline["integrity_hash"] != self._baseline_hash(baseline):
            raise RuntimeError("Baseline integrity failure.")

        if baseline["meta"]["schema_signature"] != get_schema_signature():
            raise RuntimeError("Schema mismatch with baseline.")

        return baseline

    ########################################################
    # PSI
    ########################################################

    def _psi(self, expected, actual):

        expected = np.asarray(expected)
        actual = np.asarray(actual)

        if len(expected) < 2:
            return 0.0

        bins = np.unique(expected)

        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]

        expected_perc = expected_counts / max(len(expected), self.EPSILON)
        actual_perc = actual_counts / max(len(actual), self.EPSILON)

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

        baseline = self._load_baseline()

        numeric = dataset.tail(self.MAX_INFERENCE_ROWS)

        drift_detected = False
        report = {}

        for col, stats in baseline["features"].items():

            current = numeric[col].dropna()

            if len(current) < self.MIN_SAMPLE_INFERENCE:
                continue

            z_score = abs(current.mean() - stats["mean"]) / max(
                stats["std"],
                self.EPSILON
            )

            variance_ratio = current.var() / max(
                stats["variance"],
                self.EPSILON
            )

            psi = self._psi(
                stats["distribution"],
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

        return {
            "drift_detected": drift_detected,
            "details": report
        }
