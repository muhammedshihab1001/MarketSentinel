import numpy as np
import pandas as pd
import shutil
import os
import json
import pytest

from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import MODEL_FEATURES
from core.artifacts.metadata_manager import MetadataManager


# ======================================================
# UTILITIES
# ======================================================

def _build_random_df(n=300):
    np.random.seed(42)
    return pd.DataFrame({
        col: np.random.normal(0, 1, n)
        for col in MODEL_FEATURES
    })


def _clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


# ======================================================
# BASELINE CREATION + NORMAL DETECTION
# ======================================================

def test_drift_detector_baseline_and_detect():

    df = _build_random_df()
    test_dir = "artifacts/drift_test"

    _clean_dir(test_dir)

    detector = DriftDetector(baseline_dir=test_dir)

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

    _clean_dir(test_dir)


# ======================================================
# STRONG MEAN SHIFT DETECTION
# ======================================================

def test_drift_detector_detects_shift():

    df_train = _build_random_df()
    df_shift = _build_random_df()

    for col in MODEL_FEATURES:
        df_shift[col] += 5.0

    test_dir = "artifacts/drift_test_shift"
    _clean_dir(test_dir)

    detector = DriftDetector(baseline_dir=test_dir)

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

    _clean_dir(test_dir)


# ======================================================
# INTEGRITY FAILURE (SOFT MODE)
# ======================================================

def test_drift_detector_integrity_failure_soft():

    df = _build_random_df()
    test_dir = "artifacts/drift_test_integrity"
    _clean_dir(test_dir)

    detector = DriftDetector(baseline_dir=test_dir)

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

    baseline_path = os.path.join(test_dir, "baseline.json")

    with open(baseline_path, "r") as f:
        payload = json.load(f)

    payload["meta"]["model_version"] = "tampered"

    with open(baseline_path, "w") as f:
        json.dump(payload, f)

    result = detector.detect(df)

    assert result["drift_detected"] is True
    assert result["drift_state"] == "detector_failure"
    assert result["exposure_scale"] < 1.0

    _clean_dir(test_dir)


# ======================================================
# HARD FAIL MODE
# ======================================================

def test_drift_detector_hard_fail_mode(monkeypatch):

    df = _build_random_df()
    test_dir = "artifacts/drift_test_hard"
    _clean_dir(test_dir)

    monkeypatch.setenv("DRIFT_HARD_FAIL", "true")

    detector = DriftDetector(baseline_dir=test_dir)

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

    baseline_path = os.path.join(test_dir, "baseline.json")

    with open(baseline_path, "w") as f:
        f.write("corrupted")

    with pytest.raises(Exception):
        detector.detect(df)

    _clean_dir(test_dir)


# ======================================================
# SEVERITY ESCALATION LOGIC
# ======================================================

def test_severity_escalation_levels():

    df_train = _build_random_df()
    df_mild = _build_random_df()
    df_extreme = _build_random_df()

    for col in MODEL_FEATURES:
        df_mild[col] += 1.0
        df_extreme[col] += 8.0

    test_dir = "artifacts/drift_test_severity"
    _clean_dir(test_dir)

    detector = DriftDetector(baseline_dir=test_dir)

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

    _clean_dir(test_dir)