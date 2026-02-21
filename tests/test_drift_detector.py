import numpy as np
import pandas as pd

from core.monitoring.drift_detector import DriftDetector
from core.schema.feature_schema import MODEL_FEATURES


def test_drift_detector_runs():

    n = 250  # must exceed MIN_SAMPLE_BASELINE

    df = pd.DataFrame({
        col: np.random.normal(0, 1, n)
        for col in MODEL_FEATURES
    })

    detector = DriftDetector()

    detector.create_baseline(
        dataset=df,
        dataset_hash="testhash",
        training_code_hash="codehash",
        allow_overwrite=True
    )

    result = detector.detect(df)

    assert "drift_detected" in result