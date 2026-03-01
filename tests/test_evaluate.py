import numpy as np
import pandas as pd

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# REGRESSION-SAFE DUMMY TRAINER
############################################################

def dummy_trainer(train_df):

    class DummyModel:
        def predict(self, X):
            # deterministic regression-safe signal
            return X.iloc[:, 0].values.astype("float32")

    return DummyModel()


############################################################
# HELPER DATA
############################################################

def build_dataset(n_days=400, n_tickers=20, seed=42):

    np.random.seed(seed)

    dates = pd.date_range("2024-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]

    rows = []

    for d in dates:
        for t in tickers:
            row = {
                "date": d,
                "ticker": t,
                "close": 100 + np.random.randn(),
                "volatility": abs(np.random.randn()) + 0.1
            }

            # Add all required model features
            for col in MODEL_FEATURES:
                row[col] = np.random.randn()

            rows.append(row)

    return pd.DataFrame(rows)


############################################################
# METRIC SANITY
############################################################

def test_metrics_computation():

    df = build_dataset()

    wf = WalkForwardValidator(dummy_trainer)

    metrics = wf.run(df)

    assert np.isfinite(metrics["avg_sharpe"])
    assert np.isfinite(metrics["profit_factor"])
    assert metrics["final_equity"] > 0


############################################################
# SHARPE GATE BOUNDS
############################################################

def test_sharpe_bounds():

    df = build_dataset()

    wf = WalkForwardValidator(dummy_trainer)

    metrics = wf.run(df)

    assert -5.0 <= metrics["avg_sharpe"] <= 5.0


############################################################
# DRAWDOWN SAFETY
############################################################

def test_drawdown_bounds():

    df = build_dataset()

    wf = WalkForwardValidator(dummy_trainer)

    metrics = wf.run(df)

    assert -1.0 <= metrics["max_drawdown"] <= 0.0


############################################################
# METRIC TYPE STABILITY
############################################################

def test_metric_type_stability():

    df = build_dataset()

    wf = WalkForwardValidator(dummy_trainer)

    metrics = wf.run(df)

    for v in metrics.values():
        assert isinstance(v, (int, float))