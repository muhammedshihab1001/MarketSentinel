import numpy as np
import pandas as pd

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# STRONG DISPERSION DUMMY TRAINER
############################################################

def dummy_trainer(train_df):

    class DummyModel:
        def predict(self, X):
            # strong deterministic cross-sectional signal
            base = X.iloc[:, 0].values
            noise = np.linspace(-1, 1, len(base))
            return (base * 0.5 + noise).astype("float32")

    return DummyModel()


############################################################
# END-TO-END WALKFORWARD
############################################################

def test_walkforward_runs_end_to_end():

    np.random.seed(42)

    dates = pd.date_range("2026-01-01", periods=500)
    tickers = ["A","B","C","D","E","F","G","H","I","J","K","L"]

    rows = []

    for d in dates:
        for t in tickers:
            row = {
                "date": d,
                "ticker": t,
                "close": 100 + np.random.randn(),
                "volatility": abs(np.random.randn()) + 0.1
            }

            for col in MODEL_FEATURES:
                row[col] = np.random.randn()

            rows.append(row)

    df = pd.DataFrame(rows)

    validator = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    metrics = validator.run(df)

    assert isinstance(metrics, dict)
    assert metrics["num_windows"] > 0
    assert metrics["final_equity"] > 0
    assert -5.0 <= metrics["avg_sharpe"] <= 5.0