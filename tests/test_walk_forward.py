import pandas as pd
import numpy as np

from training.backtesting.walk_forward import WalkForwardValidator


def dummy_trainer(df):
    class Dummy:
        def predict_proba(self, X):
            return np.tile([0.4, 0.6], (len(X), 1))
    return Dummy()


def dummy_signal(model, df):
    return ["HOLD"] * len(df)


def test_walk_forward_requires_enough_data():

    df = pd.DataFrame({
        "date": pd.date_range("2020", periods=300),
        "close": np.random.rand(300)
    })

    wf = WalkForwardValidator(
        dummy_trainer,
        dummy_signal,
        window_size=250
    )

    try:
        wf.run(df)
        assert False
    except RuntimeError:
        assert True
