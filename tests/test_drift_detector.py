import numpy as np
import pandas as pd
import json
import pytest
import os

from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# SYNTHETIC DATA BUILDER
############################################################

def make_dataset(rows=300, seed=42, noise_scale=1.0):
    """Build a synthetic feature DataFrame matching MODEL_FEATURES schema."""
    rng = np.random.default_rng(seed)
    data = {}
    for f in MODEL_FEATURES:
        data[f] = rng.normal(0, noise_scale, rows).astype(np.float32)
    df = pd.DataFrame(data)
    df["ticker"] = "TEST"
    df["date"] = pd.date_range("2024-01-01", periods=rows, freq="B")
    return df


############################################################
# BASELINE CREATION
############################################################

class TestBaselineCreation:

    def test_create_baseline_succeeds(self, tmp_path):
        detector = DriftDetector(baseline_dir=str(tmp_path))
        df = make_dataset()
        path = detector.create_baseline(
            dataset=df,
            dataset_hash="abc" * 22,
            training_code_hash="def" * 22,
            feature_checksum="ghi" * 22,
            model_version="test_v1",
        )
        assert os.path.exists(path)

    def test_create_baseline_has_integrity_hash(self, tmp_path):
        detector = DriftDetector(baseline_dir=str(tmp_path))
        df = make_dataset()
        path = detector.create_baseline(
            dataset=df,
            dataset_hash="abc" * 22,
            training_code_hash="def" * 22,
            feature_checksum="ghi" * 22,
            model_version="test_v1",
        )
        with open(path) as f:
            payload = json.load(f)
        assert "integrity_hash" in payload
        assert len(payload["integrity_hash"]) == 64  # SHA256

    def test_create_baseline_minimum_features(self, tmp_path):
        detector = DriftDetector(baseline_dir=str(tmp_path))
        df = make_dataset()
        path = detector.create_baseline(
            dataset=df,
            dataset_hash="abc" * 22,
            training_code_hash="def" * 22,
            feature_checksum="ghi" * 22,
            model_version="test_v1",
        )
        with open(path) as f:
            payload = json.load(f)
        assert len(payload["features"]) >= detector.MIN_BASELINE_FEATURES

    def test_create_baseline_overwrite_raises_without_flag(self, tmp_path):
        detector = DriftDetector(baseline_dir=str(tmp_path))
        df = make_dataset()
        kwargs = dict(
            dataset_hash="abc" * 22,
            training_code_hash="def" * 22,
            feature_checksum="ghi" * 22,
            model_version="test_v1",
        )
        detector.create_baseline(dataset=df, **kwargs)
        with pytest.raises(RuntimeError, match="already exists"):
            detector.create_baseline(dataset=df, **kwargs)

    def test_create_baseline_overwrite_allowed_with_flag(self, tmp_path):
        detector = DriftDetector(baseline_dir=str(tmp_path))
        df = make_dataset()
        kwargs = dict(
            dataset_hash="abc" * 22,
            training_code_hash="def" * 22,
            feature_checksum="ghi" * 22,
            model_version="test_v1",
        )
        detector.create_baseline(dataset=df, **kwargs)
        # Should not raise
        detector.create_baseline(dataset=df, allow_overwrite=True, **kwargs)


############################################################
# DRIFT DETECTION
############################################################

class TestDriftDetection:

    def _make_detector_with_baseline(self, tmp_path, seed=42):
        detector = DriftDetector(baseline_dir=str(tmp_path))
        df = make_dataset(rows=400, seed=seed)
        detector.create_baseline(
            dataset=df,
            dataset_hash="abc" * 22,
            training_code_hash="def" * 22,
            feature_checksum="ghi" * 22,
            model_version="test_v1",
        )
        return detector, df

    def test_detect_returns_dict(self, tmp_path):
        detector, df = self._make_detector_with_baseline(tmp_path)
        result = detector.detect(df)
        assert isinstance(result, dict)

    def test_detect_required_fields(self, tmp_path):
        detector, df = self._make_detector_with_baseline(tmp_path)
        result = detector.detect(df)
        required = {
            "drift_detected", "severity_score", "drift_state",
            "drift_confidence", "exposure_scale", "coverage",
        }
        assert required.issubset(result.keys())

    def test_detect_no_drift_on_same_distribution(self, tmp_path):
        """Same distribution as baseline → no drift."""
        detector, df = self._make_detector_with_baseline(tmp_path, seed=42)
        result = detector.detect(df)
        assert result["severity_score"] <= detector.SOFT_SEVERITY_THRESHOLD

    def test_detect_drift_on_shifted_distribution(self, tmp_path):
        """Heavily shifted distribution → drift detected."""
        detector, _ = self._make_detector_with_baseline(tmp_path, seed=42)
        # Shift mean by 10 sigma
        shifted = make_dataset(rows=300, seed=99, noise_scale=1.0)
        for f in MODEL_FEATURES:
            shifted[f] = shifted[f] + 10.0
        result = detector.detect(shifted)
        assert result["drift_detected"] is True
        assert result["severity_score"] > 0

    def test_detect_drift_state_valid_values(self, tmp_path):
        detector, df = self._make_detector_with_baseline(tmp_path)
        result = detector.detect(df)
        valid_states = {"none", "soft", "hard", "baseline_missing", "detector_failure"}
        assert result["drift_state"] in valid_states

    def test_detect_exposure_scale_in_range(self, tmp_path):
        detector, df = self._make_detector_with_baseline(tmp_path)
        result = detector.detect(df)
        assert 0.0 <= result["exposure_scale"] <= 1.0

    def test_detect_missing_baseline_returns_safe_default(self, tmp_path):
        """Missing baseline → returns baseline_missing state, does not crash."""
        detector = DriftDetector(baseline_dir=str(tmp_path))
        df = make_dataset()
        result = detector.detect(df)
        assert result["drift_state"] == "baseline_missing"
        assert result["drift_detected"] is False

    def test_detect_infer_objects_no_future_warning(self, tmp_path):
        """
        FIX: _safe_feature_block uses .infer_objects(copy=False)
        to suppress FutureWarning from pandas clip.
        """
        import warnings
        detector, df = self._make_detector_with_baseline(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            # Should NOT raise FutureWarning after the fix
            result = detector.detect(df)
        assert isinstance(result, dict)


############################################################
# INTEGRITY
############################################################

class TestBaselineIntegrity:

    def test_tampered_baseline_raises(self, tmp_path):
        detector = DriftDetector(baseline_dir=str(tmp_path))
        df = make_dataset()
        path = detector.create_baseline(
            dataset=df,
            dataset_hash="abc" * 22,
            training_code_hash="def" * 22,
            feature_checksum="ghi" * 22,
            model_version="test_v1",
        )
        # Tamper
        with open(path) as f:
            payload = json.load(f)
        payload["meta"]["model_version"] = "tampered"
        with open(path, "w") as f:
            json.dump(payload, f)

        with pytest.raises(Exception, match="integrity"):
            detector._load_verified_baseline()

    def test_health_returns_baseline_exists(self, tmp_path):
        detector = DriftDetector(baseline_dir=str(tmp_path))
        assert detector.health()["baseline_exists"] is False

        df = make_dataset()
        detector.create_baseline(
            dataset=df,
            dataset_hash="abc" * 22,
            training_code_hash="def" * 22,
            feature_checksum="ghi" * 22,
            model_version="test_v1",
        )
        assert detector.health()["baseline_exists"] is True
