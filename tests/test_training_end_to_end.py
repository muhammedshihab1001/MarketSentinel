import numpy as np
import pandas as pd

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    DTYPE
)


def dummy_trainer(train_df):

    class DummyModel:
        def predict(self, X):
            # deterministic dispersion from first feature
            base = X.iloc[:, 0].values.astype("float32")
            return (base * 2.0)

    return DummyModel()


def test_walkforward_runs_end_to_end():

    np.random.seed(42)

    dates = pd.date_range("2026-01-01", periods=500)
    tickers = [f"T{i}" for i in range(15)]  # >= MIN_CS_WIDTH

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

    # Ensure correct dtype + schema compliance
    feature_block = validate_feature_schema(
        df.loc[:, MODEL_FEATURES],
        mode="training"
    ).astype(DTYPE)

    df.loc[:, MODEL_FEATURES] = feature_block

    validator = WalkForwardValidator(
        model_trainer=dummy_trainer,
        window_size=120,
        step_size=30
    )

    metrics = validator.run(df)

    # ======================================================
    # Core assertions
    # ======================================================

    assert metrics["num_windows"] > 0
    assert metrics["final_equity"] > 0
    assert -5.0 <= metrics["avg_sharpe"] <= 5.0

    # ======================================================
    # Additional robustness checks
    # ======================================================

    assert np.isfinite(metrics["avg_sharpe"])
    assert np.isfinite(metrics["final_equity"])
    assert metrics["profit_factor"] >= 0