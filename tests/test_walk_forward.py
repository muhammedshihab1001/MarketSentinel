import pandas as pd
import numpy as np
import pytest

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# DUMMY TRAINER
############################################################

def dummy_trainer(train_df):

    class DummyModel:
        def predict_proba(self, X):
            return np.tile([0.4, 0.6], (len(X), 1))

    return DummyModel()


############################################################
# HELPER TO BUILD VALID DATASET (PRODUCTION-LIKE)
############################################################

def build_valid_dataset(
    rows=2000,
    n_tickers=20,
    seed=42
):

    np.random.seed(seed)

    dates = pd.date_range("2020-01-01", periods=rows)
    tickers = [f"T{i}" for i in range(n_tickers)]

    df = pd.DataFrame({
        "date": np.repeat(dates, n_tickers),
        "ticker": np.tile(tickers, rows)
    })

    total_rows = len(df)

    # --- REQUIRED PRICE COLUMNS ---
    base_price = 100 + np.cumsum(np.random.randn(total_rows) * 0.5)
    df["close"] = base_price.astype("float32")

    # Volatility required for weighting
    df["volatility"] = (
        np.abs(np.random.randn(total_rows)) + 0.01
    ).astype("float32")

    # --- MODEL FEATURES ---
    for col in MODEL_FEATURES:
        df[col] = np.random.randn(total_rows).astype("float32")

    df["target"] = np.random.randint(0, 2, size=total_rows)

    return df


############################################################
# TEST INSUFFICIENT DATA
############################################################

def test_walk_forward_requires_enough_data():

    df = build_valid_dataset(rows=50, n_tickers=5)

    wf = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    with pytest.raises(RuntimeError):
        wf.run(df)


############################################################
# TEST SUCCESSFUL RUN
############################################################

def test_walk_forward_runs_successfully():

    df = build_valid_dataset(rows=2000, n_tickers=20)

    wf = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    metrics = wf.run(df)

    assert isinstance(metrics, dict)
    assert "avg_strategy_return" in metrics
    assert "num_windows" in metrics
    assert metrics["num_windows"] > 0