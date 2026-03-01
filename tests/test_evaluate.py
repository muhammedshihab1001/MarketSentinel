import numpy as np
import pytest

from training.backtesting.walk_forward import WalkForwardValidator


############################################################
# DUMMY TRAINER FOR RANKING
############################################################

def dummy_trainer(train_df):

    class DummyModel:

        def predict(self, X):
            # simple ranking variation
            base = X.iloc[:, 0].values
            return base + np.random.normal(0, 0.01, len(base))

    return DummyModel()


############################################################
# HELPER DATA
############################################################

def build_dataset(n_days=400, n_tickers=20):

    dates = np.arange(n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]

    data = []

    for d in dates:
        for t in tickers:
            data.append({
                "date": f"2020-01-{(d % 28) + 1:02d}",
                "ticker": t,
                "close": 100 + np.random.randn(),
                "volatility": 0.2,
                "momentum_20_z": np.random.randn()
            })

    return data


############################################################
# RANKING METRIC SANITY
############################################################

def test_ranking_metrics_computation():

    df = build_dataset()
    import pandas as pd
    df = pd.DataFrame(df)

    wf = WalkForwardValidator(dummy_trainer)

    metrics = wf.run(df)

    assert np.isfinite(metrics["avg_sharpe"])
    assert np.isfinite(metrics["profit_factor"])
    assert metrics["final_equity"] > 0


############################################################
# GATE ENFORCEMENT (SHARPE FLOOR)
############################################################

def test_sharpe_gate_sanity():

    df = build_dataset()
    import pandas as pd
    df = pd.DataFrame(df)

    wf = WalkForwardValidator(dummy_trainer)

    metrics = wf.run(df)

    # Sharpe must be within clipped institutional bounds
    assert -5.0 <= metrics["avg_sharpe"] <= 5.0


############################################################
# DRAWDOWN SAFETY
############################################################

def test_drawdown_bounds():

    df = build_dataset()
    import pandas as pd
    df = pd.DataFrame(df)

    wf = WalkForwardValidator(dummy_trainer)

    metrics = wf.run(df)

    # drawdown must be between -1 and 0
    assert -1.0 <= metrics["max_drawdown"] <= 0.0


############################################################
# METRIC TYPE STABILITY
############################################################

def test_metric_type_stability():

    df = build_dataset()
    import pandas as pd
    df = pd.DataFrame(df)

    wf = WalkForwardValidator(dummy_trainer)

    metrics = wf.run(df)

    for v in metrics.values():
        assert isinstance(v, (int, float))