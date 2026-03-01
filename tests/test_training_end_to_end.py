import pandas as pd
import numpy as np

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# REGRESSION-SAFE DUMMY TRAINER
############################################################

def dummy_trainer(train_df):

    class DummyModel:
        def predict(self, X):
            # deterministic signal based on first feature
            return X.iloc[:, 0].values.astype("float32")

    return DummyModel()


############################################################
# END-TO-END WALKFORWARD TEST
############################################################

def test_walkforward_runs_end_to_end():

    np.random.seed(42)

    dates = pd.date_range("2026-01-01", periods=500)
    tickers = ["A","B","C","D","E","F","G","H","I","J"]

    rows = []

    for d in dates:
        for t in tickers:
            row = {
                "date": d,
                "ticker": t,
                "close": 100 + np.random.randn(),
                "volatility": abs(np.random.randn()) + 0.1
            }

            # add required model features
            for col in MODEL_FEATURES:
                row[col] = np.random.randn()

            rows.append(row)

    df = pd.DataFrame(rows)

    validator = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    metrics = validator.run(df)

    # Validate structure
    assert isinstance(metrics, dict)
    assert metrics["num_windows"] > 0
    assert np.isfinite(metrics["avg_sharpe"])
    assert np.isfinite(metrics["final_equity"])
    assert metrics["final_equity"] > 0