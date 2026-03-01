import pandas as pd
import numpy as np
import pytest

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# DUMMY TRAINER (RANKING-COMPATIBLE)
############################################################

def dummy_trainer(train_df):

    class DummyModel:

        def predict(self, X):
            # create deterministic cross-sectional variation
            base = X.iloc[:, 0].values
            return base + np.random.normal(0, 0.01, len(base))

    return DummyModel()


############################################################
# HELPER TO BUILD VALID DATASET
############################################################

def build_valid_dataset(
    n_days=400,
    n_tickers=20,
    seed=42
):

    np.random.seed(seed)

    dates = pd.date_range("2020-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]

    df = pd.DataFrame({
        "date": np.repeat(dates, n_tickers),
        "ticker": np.tile(tickers, n_days)
    })

    total_rows = len(df)

    # --- PRICE SERIES ---
    df["close"] = (
        100 + np.cumsum(np.random.randn(total_rows) * 0.5)
    ).astype("float32")

    # --- VOLATILITY REQUIRED FOR PORTFOLIO CONSTRUCTION ---
    df["volatility"] = (
        np.abs(np.random.randn(total_rows)) + 0.05
    ).astype("float32")

    # --- REQUIRED MODEL FEATURES ---
    for col in MODEL_FEATURES:
        df[col] = np.random.randn(total_rows).astype("float32")

    return df


############################################################
# TEST INSUFFICIENT DATA
############################################################

def test_walk_forward_requires_enough_data():

    df = build_valid_dataset(n_days=50, n_tickers=5)

    wf = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    with pytest.raises(Exception):
        wf.run(df)


############################################################
# TEST SUCCESSFUL RUN
############################################################

def test_walk_forward_runs_successfully():

    df = build_valid_dataset(n_days=400, n_tickers=20)

    wf = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    metrics = wf.run(df)

    required_keys = {
        "avg_strategy_return",
        "avg_sharpe",
        "profit_factor",
        "max_drawdown",
        "return_volatility",
        "final_equity",
        "avg_turnover",
        "avg_win_rate",
        "avg_trades_per_window",
        "num_windows"
    }

    assert isinstance(metrics, dict)
    assert required_keys.issubset(metrics.keys())
    assert metrics["num_windows"] > 0
    assert np.isfinite(metrics["avg_sharpe"])
    assert metrics["final_equity"] > 0


############################################################
# TEST CROSS-SECTIONAL WIDTH ENFORCEMENT
############################################################

def test_walk_forward_requires_cross_section():

    df = build_valid_dataset(n_days=400, n_tickers=5)  # below MIN_CROSS_SECTION

    wf = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    with pytest.raises(Exception):
        wf.run(df)


############################################################
# TEST STABILITY UNDER RANDOM DATA
############################################################

def test_walk_forward_stability():

    df = build_valid_dataset(n_days=450, n_tickers=25)

    wf = WalkForwardValidator(
        model_trainer=dummy_trainer
    )

    metrics = wf.run(df)

    # Ensure metrics within sane bounds
    assert -5.0 <= metrics["avg_sharpe"] <= 5.0
    assert -1.0 <= metrics["max_drawdown"] <= 0.0