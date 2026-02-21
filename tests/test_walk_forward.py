import pandas as pd
import numpy as np
import pytest

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# DUMMY TRAINER (Aligned With Current Architecture)
############################################################

def dummy_trainer(train_df):

    class DummyModel:
        def predict_proba(self, X):
            # constant 60% up probability
            return np.tile([0.4, 0.6], (len(X), 1))

    return DummyModel()


############################################################
# HELPER TO BUILD VALID DATASET
############################################################

def build_valid_dataset(rows=500):

    dates = pd.date_range("2020-01-01", periods=rows)

    df = pd.DataFrame({
        "date": np.tile(dates, 1),
        "ticker": "TEST"
    })

    for col in MODEL_FEATURES:
        df[col] = np.random.randn(rows).astype("float32")

    df["target"] = np.random.randint(0, 2, size=rows)

    return df


############################################################
# TEST INSUFFICIENT DATA
############################################################

def test_walk_forward_requires_enough_data():

    df = build_valid_dataset(rows=100)

    wf = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    with pytest.raises(RuntimeError):
        wf.run(df)


############################################################
# TEST SUCCESSFUL RUN
############################################################

def test_walk_forward_runs_successfully():

    df = build_valid_dataset(rows=600)

    wf = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    metrics = wf.run(df)

    assert isinstance(metrics, dict)
    assert "avg_strategy_return" in metrics
    assert "num_windows" in metrics
    assert metrics["num_windows"] > 0