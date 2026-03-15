# =========================================================
# DRIFT DETECTOR v2.1
# Hybrid Multi-Agent Compatible | CV-Optimized
# Noise-Tolerant for yfinance data
# =========================================================

import numpy as np
import pandas as pd
import os
import json
import logging
import hashlib

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES,
    DTYPE
)

from core.artifacts.metadata_manager import MetadataManager

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
    Lightweight drift detector designed for noisy financial data.

    Signals:
    - z-score mean shift
    - variance shift
    - PSI distribution drift

    Designed for CV showcase and yfinance robustness.
    """

    BASELINE_FILENAME = "baseline.json"
    BASELINE_VERSION = "26.0"

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

    FEATURE_CLIP_SIGMA = 6.0  # NEW: protects against Yahoo spikes

    # -----------------------------------------------------

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
    # SAFE FEATURE BLOCK
    # =====================================================

    def _safe_feature_block(self, dataset: pd.DataFrame):

        missing = set(MODEL_FEATURES) - set(dataset.columns)

        if missing:
            raise RuntimeError(f"Missing features: {missing}")

        block = dataset.loc[:, MODEL_FEATURES].copy()

        for col in MODEL_FEATURES:

            block[col] = pd.to_numeric(block[col], errors="coerce").astype(DTYPE)

            block[col] = block[col].replace([np.inf, -np.inf], np.nan)

            # NEW: clip extreme spikes
            mean = block[col].mean()
            std = block[col].std()

            if std > 0:
                block[col] = block[col].clip(
                    mean - self.FEATURE_CLIP_SIGMA * std,
                    mean + self.FEATURE_CLIP_SIGMA * std
                )

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
    # LOAD VERIFIED BASELINE
    # =====================================================

    def _load_verified_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):
            raise FileNotFoundError("Baseline missing.")

        with open(self.BASELINE_PATH, encoding="utf-8") as f:
            baseline = json.load(f)

        if baseline.get("integrity_hash") != self._baseline_hash(baseline):
            raise RuntimeError("Baseline integrity failure.")

        meta = baseline.get("meta", {})

        if meta.get("baseline_version") != self.BASELINE_VERSION:
            logger.warning("Baseline version mismatch.")

        if meta.get("schema_signature") != get_schema_signature():
            logger.warning("Schema signature mismatch.")

        current_checksum = MetadataManager.fingerprint_features(
            tuple(MODEL_FEATURES)
        )

        if meta.get("feature_checksum") != current_checksum:
            logger.warning("Feature checksum mismatch.")

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
                int(np.sqrt(severity_accumulator)),  # NEW: smoother scaling
                self.MAX_SEVERITY_CAP
            )

            drift_state = (
                "hard" if severity_score >= self.HARD_SEVERITY_THRESHOLD
                else "soft" if severity_score >= self.SOFT_SEVERITY_THRESHOLD
                else "none"
            )

            drift_detected = drift_count > 0

            DRIFT_DETECTED.set(severity_score)

            if drift_state == "hard":

                exposure_scale = 0.2

            elif drift_state == "soft":

                exposure_scale = max(0.4, 1 - severity_score * 0.04)

            else:

                exposure_scale = 1.0

            return {
                "drift_detected": drift_detected,
                "severity_score": severity_score,
                "drift_confidence": float(
                    min(drift_count / max(total_features, 1), 1.0)
                ),
                "coverage": coverage,
                "details": report,
                "drift_state": drift_state,
                "exposure_scale": float(np.clip(exposure_scale, 0.0, 1.0)),
                "drift_summary": {
                    "evaluated_features": evaluated_features,
                    "total_features": total_features
                }
            }

        except FileNotFoundError:

            logger.warning("Drift baseline missing.")

            return {
                "drift_detected": False,
                "severity_score": 0,
                "drift_confidence": 0.0,
                "coverage": 0.0,
                "details": {},
                "drift_state": "baseline_missing",
                "exposure_scale": 1.0,
                "reason": "baseline_missing"
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
                "coverage": 0.0,
                "details": {},
                "drift_state": "detector_failure",
                "exposure_scale": 0.4,
                "reason": "detector_failure"
            }