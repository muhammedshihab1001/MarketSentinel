import numpy as np
import pandas as pd
import json
import pytest

from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import MODEL_FEATURES
from core.artifacts.metadata_manager import MetadataManager


############################################################
# UTILITIES
############################################################

def _build_random_df(n=300):

    rng = np.random.default_rng(42)

    return pd.DataFrame({
        col: rng.normal(0, 1, n)
        for col in MODEL_FEATURES
    })


############################################################
# BASELINE CREATION + NORMAL DETECTION
############################################################

def test_drift_detector_baseline_and_detect(tmp_path):

    df = _build_random_df()

    detector = DriftDetector(baseline_dir=str(tmp_path))

    detector.create_baseline(
        dataset=df,
        dataset_hash="testhash",
        training_code_hash="codehash",
        feature_checksum=MetadataManager.fingerprint_features(
            tuple(MODEL_FEATURES)
        ),
        model_version="test_version",
        allow_overwrite=True
    )

    result = detector.detect(df)

    required_fields = {
        "drift_detected",
        "severity_score",
        "drift_confidence",
        "coverage",
        "cross_sectional_stability",
        "details",
        "drift_state",
        "exposure_scale"
    }

    assert required_fields.issubset(result.keys())

    assert np.isfinite(result["severity_score"])
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["exposure_scale"] in {0.0, 0.5, 1.0}
    assert isinstance(result["details"], dict)


############################################################
# STRONG MEAN SHIFT DETECTION
############################################################

def test_drift_detector_detects_shift(tmp_path):

    df_train = _build_random_df()
    df_shift = _build_random_df()

    for col in MODEL_FEATURES:
        df_shift[col] += 5.0

    detector = DriftDetector(baseline_dir=str(tmp_path))

    detector.create_baseline(
        dataset=df_train,
        dataset_hash="hash",
        training_code_hash="codehash",
        feature_checksum=MetadataManager.fingerprint_features(
            tuple(MODEL_FEATURES)
        ),
        model_version="v1",
        allow_overwrite=True
    )

    result = detector.detect(df_shift)

    assert result["drift_detected"] is True
    assert result["severity_score"] > 0
    assert result["drift_state"] in {"soft", "hard"}
    assert result["exposure_scale"] in {0.0, 0.5}


############################################################
# INTEGRITY FAILURE (SOFT MODE)
############################################################

def test_drift_detector_integrity_failure_soft(tmp_path):

    df = _build_random_df()

    detector = DriftDetector(baseline_dir=str(tmp_path))

    detector.create_baseline(
        dataset=df,
        dataset_hash="hash",
        training_code_hash="codehash",
        feature_checksum=MetadataManager.fingerprint_features(
            tuple(MODEL_FEATURES)
        ),
        model_version="v1",
        allow_overwrite=True
    )

    baseline_path = tmp_path / "baseline.json"

    with open(baseline_path, "r") as f:
        payload = json.load(f)

    payload["meta"]["model_version"] = "tampered"

    with open(baseline_path, "w") as f:
        json.dump(payload, f)

    result = detector.detect(df)

    assert result["drift_detected"] is True
    assert result["drift_state"] == "detector_failure"
    assert result["exposure_scale"] < 1.0


############################################################
# HARD FAIL MODE
############################################################

def test_drift_detector_hard_fail_mode(monkeypatch, tmp_path):

    df = _build_random_df()

    monkeypatch.setenv("DRIFT_HARD_FAIL", "true")

    detector = DriftDetector(baseline_dir=str(tmp_path))

    detector.create_baseline(
        dataset=df,
        dataset_hash="hash",
        training_code_hash="codehash",
        feature_checksum=MetadataManager.fingerprint_features(
            tuple(MODEL_FEATURES)
        ),
        model_version="v1",
        allow_overwrite=True
    )

    baseline_path = tmp_path / "baseline.json"

    with open(baseline_path, "w") as f:
        f.write("corrupted")

    with pytest.raises(Exception):
        detector.detect(df)


############################################################
# SEVERITY ESCALATION LOGIC
############################################################

def test_severity_escalation_levels(tmp_path):

    df_train = _build_random_df()
    df_mild = _build_random_df()
    df_extreme = _build_random_df()

    for col in MODEL_FEATURES:
        df_mild[col] += 1.0
        df_extreme[col] += 8.0

    detector = DriftDetector(baseline_dir=str(tmp_path))

    detector.create_baseline(
        dataset=df_train,
        dataset_hash="hash",
        training_code_hash="codehash",
        feature_checksum=MetadataManager.fingerprint_features(
            tuple(MODEL_FEATURES)
        ),
        model_version="v1",
        allow_overwrite=True
    )

    mild_result = detector.detect(df_mild)
    extreme_result = detector.detect(df_extreme)

    assert extreme_result["severity_score"] >= mild_result["severity_score"]