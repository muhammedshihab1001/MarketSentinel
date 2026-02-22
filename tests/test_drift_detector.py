import numpy as np
import pandas as pd
import shutil
import os

from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import MODEL_FEATURES
from core.artifacts.metadata_manager import MetadataManager


def test_drift_detector_runs():

    n = 250

    df = pd.DataFrame({
        col: np.random.normal(0, 1, n)
        for col in MODEL_FEATURES
    })

    test_dir = "artifacts/drift_test"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

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

    assert "drift_detected" in result

    shutil.rmtree(test_dir)