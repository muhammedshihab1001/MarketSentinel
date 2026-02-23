import numpy as np
import pandas as pd
import shutil
import os
import json
import pytest

from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import MODEL_FEATURES
from core.artifacts.metadata_manager import MetadataManager


def _build_random_df(n=250):
    return pd.DataFrame({
        col: np.random.normal(0, 1, n)
        for col in MODEL_FEATURES
    })


def test_drift_detector_baseline_and_detect():

    n = 250
    df = _build_random_df(n)

    test_dir = "artifacts/drift_test"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    detector = DriftDetector(baseline_dir=test_dir)

    # Create baseline
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

    # Detect on same distribution (should not drift heavily)
    result = detector.detect(df)

    assert isinstance(result, dict)
    assert "drift_detected" in result
    assert "severity_score" in result
    assert "coverage" in result
    assert "details" in result

    assert result["coverage"] > 0
    assert result["severity_score"] >= 0

    shutil.rmtree(test_dir)


def test_drift_detector_detects_shift():

    df_train = _build_random_df(250)
    df_shift = _build_random_df(250)

    # Inject strong mean shift
    for col in MODEL_FEATURES:
        df_shift[col] += 5.0

    test_dir = "artifacts/drift_test_shift"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

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

    shutil.rmtree(test_dir)


def test_drift_detector_integrity_failure():

    df = _build_random_df(250)
    test_dir = "artifacts/drift_test_integrity"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

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

    # Tamper with baseline file
    baseline_path = os.path.join(test_dir, "baseline.json")

    with open(baseline_path, "r") as f:
        payload = json.load(f)

    payload["meta"]["model_version"] = "tampered"

    with open(baseline_path, "w") as f:
        json.dump(payload, f)

    result = detector.detect(df)

    # Should fall into detector_failure branch (soft mode)
    assert result["drift_detected"] is True
    assert result.get("reason") == "detector_failure"

    shutil.rmtree(test_dir)


def test_drift_detector_hard_fail_mode():

    df = _build_random_df(250)
    test_dir = "artifacts/drift_test_hard"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.environ["DRIFT_HARD_FAIL"] = "true"

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

    # Corrupt baseline intentionally
    baseline_path = os.path.join(test_dir, "baseline.json")
    with open(baseline_path, "w") as f:
        f.write("corrupted")

    with pytest.raises(Exception):
        detector.detect(df)

    shutil.rmtree(test_dir)
    os.environ["DRIFT_HARD_FAIL"] = "false"