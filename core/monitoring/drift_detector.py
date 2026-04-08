# =========================================================
# DRIFT DETECTOR v2.7
# Hybrid Multi-Agent Compatible | CV-Optimized
# Noise-Tolerant for yfinance data
# =========================================================

import numpy as np
import pandas as pd
import os
import json

pd.set_option("future.no_silent_downcasting", True)
import logging
import hashlib

from core.schema.feature_schema import (
    get_schema_signature,
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

    BASELINE_FILENAME = "baseline.json"
    BASELINE_VERSION = "26.3"

    MIN_SAMPLE_INFERENCE = 25
    MIN_FEATURE_EVAL_RATIO = 0.3

    VARIANCE_RATIO_UPPER = 4.0
    VARIANCE_RATIO_LOWER = 0.25

    PSI_ALERT = 0.45
    MIN_BIN_PCT = 0.01

    MAX_INFERENCE_ROWS = 800

    EPSILON = 1e-8
    MAX_SEVERITY_CAP = 15

    SOFT_SEVERITY_THRESHOLD = 4
    HARD_SEVERITY_THRESHOLD = 10

    RECENT_WEIGHT_FACTOR = 1.5
    FEATURE_CLIP_SIGMA = 6.0

    MIN_BASELINE_FEATURES = 10

    FEATURE_SET = set(MODEL_FEATURES)

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self, z_threshold: float = 4.0, baseline_dir: str = "artifacts/drift"):

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

    # =====================================================
    # BASELINE CREATION
    # =====================================================

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
            raise RuntimeError("Cannot create baseline from empty dataset.")

        if os.path.exists(self.BASELINE_PATH) and not allow_overwrite:
            raise RuntimeError("Baseline already exists.")

        numeric = self._safe_feature_block(dataset)

        features = {}

        for col in MODEL_FEATURES:

            if col not in numeric.columns:
                continue

            series = numeric[col].dropna()

            if len(series) < 20:
                continue

            counts, bin_edges = np.histogram(series, bins=20)

            if counts.sum() == 0:
                continue

            features[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "variance": float(series.var()),
                "bin_edges": bin_edges.tolist(),
                "expected_counts": counts.tolist()
            }

        if len(features) < self.MIN_BASELINE_FEATURES:
            raise RuntimeError(
                "Insufficient baseline features detected."
            )

        payload = {
            "meta": {
                "baseline_version": self.BASELINE_VERSION,
                "schema_signature": get_schema_signature(),
                "feature_checksum": feature_checksum,
                "dataset_hash": dataset_hash,
                "training_code_hash": training_code_hash,
                "model_version": model_version
            },
            "features": features
        }

        payload["integrity_hash"] = self._baseline_hash(payload)

        with open(self.BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Drift baseline created: %s", self.BASELINE_PATH)

        return self.BASELINE_PATH

    # =====================================================
    # SAFE FEATURE BLOCK
    # =====================================================

    def _safe_feature_block(self, dataset: pd.DataFrame):

        missing = self.FEATURE_SET - set(dataset.columns)

        if missing:
            raise RuntimeError(f"Missing features: {missing}")

        block = dataset.loc[:, MODEL_FEATURES].copy()

        for col in MODEL_FEATURES:

            block[col] = pd.to_numeric(block[col], errors="coerce").astype(DTYPE)

            block[col] = block[col].replace([np.inf, -np.inf], np.nan)

            mean = block[col].mean()
            std = block[col].std()

            if not np.isfinite(mean) or not np.isfinite(std) or std < self.EPSILON:
                continue

            block[col] = block[col].clip(
                mean - self.FEATURE_CLIP_SIGMA * std,
                mean + self.FEATURE_CLIP_SIGMA * std
            ).infer_objects(copy=False)

        return block

    # =====================================================
    # BASELINE HASH
    # =====================================================

    @staticmethod
    def _baseline_hash(payload: dict) -> str:

        clone = dict(payload)
        clone.pop("integrity_hash", None)

        canonical = json.dumps(clone, sort_keys=True).encode()

        return hashlib.sha256(canonical).hexdigest()

    # =====================================================
    # LOAD BASELINE
    # =====================================================

    def _load_verified_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise FileNotFoundError("Baseline missing.")

        try:

            with open(self.BASELINE_PATH, encoding="utf-8") as f:
                baseline = json.load(f)

        except Exception as exc:
            raise RuntimeError("Baseline corrupted.") from exc

        if baseline.get("integrity_hash") != self._baseline_hash(baseline):
            raise RuntimeError("Baseline integrity failure.")

        meta = baseline.get("meta", {})

        if meta.get("schema_signature") != get_schema_signature():
            logger.warning("Schema mismatch detected.")

        if len(baseline.get("features", {})) < self.MIN_BASELINE_FEATURES:
            logger.warning("Baseline feature count suspiciously low.")

        return baseline

    # =====================================================
    # PSI
    # =====================================================

    def _psi(self, bin_edges, expected_counts, actual):

        if len(bin_edges) < 2 or len(expected_counts) == 0:
            return 0.0

        actual = np.asarray(actual, dtype=np.float64)

        if len(actual) < 5:
            return 0.0

        try:
            actual_counts = np.histogram(actual, bins=bin_edges)[0]
        except Exception:
            return 0.0

        expected_perc = expected_counts / max(expected_counts.sum(), self.EPSILON)
        actual_perc = actual_counts / max(actual_counts.sum(), self.EPSILON)

        expected_perc = np.clip(expected_perc, self.MIN_BIN_PCT, None)
        actual_perc = np.clip(actual_perc, self.MIN_BIN_PCT, None)

        psi = np.sum(
            (actual_perc - expected_perc) *
            np.log((actual_perc + self.EPSILON) /
                   (expected_perc + self.EPSILON))
        )

        return float(max(psi, 0.0))

    # =====================================================
    # TIME WEIGHTED MEAN
    # =====================================================

    def _time_weighted_mean(self, series):

        if len(series) < 5:
            return float(series.mean())

        weights = np.linspace(1.0, self.RECENT_WEIGHT_FACTOR, len(series))
        weights /= weights.sum()

        return float(np.sum(series.values * weights))

    # =====================================================
    # EXPOSURE SCALE
    # =====================================================

    def _exposure_scale(self, drift_state):

        if drift_state == "none":
            return 1.0

        if drift_state == "soft":
            return 0.6

        if drift_state == "hard":
            return 0.25

        if drift_state == "detector_failure":
            return 0.4

        return 0.5

    # =====================================================
    # DETECT
    # =====================================================

    def detect(self, dataset: pd.DataFrame):

        try:

            baseline = self._load_verified_baseline()

            numeric = self._safe_feature_block(
                dataset.tail(self.MAX_INFERENCE_ROWS)
            )

            drift_count = 0
            severity_accumulator = 0
            report = {}

            total_features = len(baseline.get("features", {}))
            evaluated_features = 0

            for col, stats in baseline.get("features", {}).items():

                if col not in numeric.columns:
                    continue

                current = numeric[col].dropna()

                if len(current) < self.MIN_SAMPLE_INFERENCE:
                    continue

                evaluated_features += 1

                weighted_mean = self._time_weighted_mean(current)

                baseline_std = max(stats.get("std", 1.0), self.EPSILON)
                baseline_var = max(stats.get("variance", 1.0), self.EPSILON)
                current_var = max(current.var(), self.EPSILON)

                z_score = abs(weighted_mean - stats.get("mean", 0.0)) / baseline_std
                variance_ratio = current_var / baseline_var

                psi = self._psi(
                    np.array(stats.get("bin_edges", [])),
                    np.array(stats.get("expected_counts", [])),
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
                        min(z_score / self.z_threshold, 4) +
                        min(abs(np.log(variance_ratio + self.EPSILON)), 4) +
                        min(psi / self.PSI_ALERT, 4)
                    )

                report[col] = {
                    "z_score": float(z_score),
                    "variance_ratio": float(variance_ratio),
                    "psi": float(psi),
                    "drift": drift
                }

            coverage = float(evaluated_features / max(total_features, 1))

            if coverage < self.MIN_FEATURE_EVAL_RATIO:
                drift_count = 0
                severity_accumulator = 0

            severity_score = min(
                int(np.sqrt(severity_accumulator)),
                self.MAX_SEVERITY_CAP
            )

            drift_state = (
                "hard" if severity_score >= self.HARD_SEVERITY_THRESHOLD
                else "soft" if severity_score >= self.SOFT_SEVERITY_THRESHOLD
                else "none"
            )

            drift_confidence = min(1.0, severity_score / self.MAX_SEVERITY_CAP)

            exposure_scale = self._exposure_scale(drift_state)

            DRIFT_DETECTED.set(severity_score)

            return {
                "drift_detected": drift_count > 0,
                "severity_score": severity_score,
                "drift_confidence": float(drift_confidence),
                "drift_state": drift_state,
                "coverage": coverage,
                "details": report,
                "exposure_scale": exposure_scale
            }

        except FileNotFoundError:

            logger.warning("Drift baseline missing.")

            return {
                "drift_detected": False,
                "severity_score": 0,
                "drift_confidence": 0.0,
                "drift_state": "baseline_missing",
                "coverage": 0,
                "details": {},
                "exposure_scale": 1.0
            }

        except Exception as exc:

            logger.warning("Drift detector failure: %s", exc)

            if self.hard_fail:
                raise

            DRIFT_DETECTED.set(6)

            return {
                "drift_detected": True,
                "severity_score": 6,
                "drift_confidence": 0.5,
                "drift_state": "detector_failure",
                "coverage": 0,
                "details": {},
                "exposure_scale": 0.4
            }

    # =====================================================
    # BACKWARD COMPATIBILITY
    # =====================================================

    def compute_drift(self, dataset: pd.DataFrame):

        result = self.detect(dataset)

        if isinstance(result, dict):
            return result.get("severity_score", 0)

        return 0

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    def health(self):

        return {
            "baseline_exists": os.path.exists(self.BASELINE_PATH),
            "baseline_path": self.BASELINE_PATH
        }
