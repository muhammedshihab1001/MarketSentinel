import numpy as np
import pandas as pd

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import MODEL_FEATURES


def dummy_trainer(train_df):

    class DummyModel:
        def predict(self, X):
            # strong deterministic dispersion
            return (X.iloc[:, 0].values * 2.0).astype("float32")

    return DummyModel()


def test_walkforward_runs_end_to_end():

    np.random.seed(42)

    dates = pd.date_range("2026-01-01", periods=500)
    tickers = [f"T{i}" for i in range(15)]

    rows = []

    for d in dates:
        for t in tickers:

            row = {
                "date": d,
                "ticker": t,
                "close": 100 + np.random.randn() * 0.5 + (d.day * 0.01),
                "volatility": abs(np.random.randn()) + 0.2
            }

            for col in MODEL_FEATURES:
                row[col] = np.random.randn()

            rows.append(row)

    df = pd.DataFrame(rows)

    validator = WalkForwardValidator(
        model_trainer=dummy_trainer,
        window_size=120,
        step_size=30
    )

    metrics = validator.run(df)

    assert metrics["num_windows"] > 0
    assert metrics["final_equity"] > 0
    assert -5.0 <= metrics["avg_sharpe"] <= 5.0