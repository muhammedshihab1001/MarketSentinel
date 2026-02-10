import pandas as pd
import numpy as np

from core.monitoring.drift_detector import DriftDetector


def test_drift_detector_runs():

    df = pd.DataFrame({
        "return": np.random.normal(0, 1, 100),
        "volatility": np.random.normal(1, 0.1, 100),
        "rsi": np.random.uniform(30, 70, 100),
        "macd": np.random.normal(0, 1, 100),
        "macd_signal": np.random.normal(0, 1, 100),
        "avg_sentiment": np.random.normal(0, 1, 100),
        "news_count": np.random.randint(1, 10, 100),
        "sentiment_std": np.random.rand(100),
        "return_lag1": np.random.normal(0, 1, 100),
        "sentiment_lag1": np.random.normal(0, 1, 100),
    })

    detector = DriftDetector()
    detector.create_baseline(df)

    result = detector.detect(df)

    assert "drift_detected" in result
