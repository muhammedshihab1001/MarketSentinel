import numpy as np
import pandas as pd

from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    DTYPE
)


############################################################
# DUMMY MODEL TRAINER
############################################################

def dummy_trainer(train_df):

    class DummyModel:

        def predict(self, X):
            # deterministic dispersion using first feature
            base = X.iloc[:, 0].values.astype("float32")
            return base * 2.0

    return DummyModel()


############################################################
# SYNTHETIC DATASET
############################################################

def build_synthetic_dataset():

    rng = np.random.default_rng(42)

    dates = pd.date_range("2026-01-01", periods=500)
    tickers = [f"T{i}" for i in range(15)]  # >= MIN_CS_WIDTH

    rows = []

    for d in dates:
        for t in tickers:

            row = {
                "date": d,
                "ticker": t,

                # small time trend prevents flat signals
                "close": 100 + rng.normal(0, 0.5) + (d.day * 0.02),

                "volatility": abs(rng.normal()) + 0.2,

                "regime": "SIDEWAYS",
                "market_regime": "SIDEWAYS",
                "regime_multiplier": 1.0
            }

            # populate schema features
            for col in MODEL_FEATURES:
                row[col] = rng.normal()

            rows.append(row)

    df = pd.DataFrame(rows)

    # ensure schema compliance
    feature_block = validate_feature_schema(
        df.loc[:, MODEL_FEATURES],
        mode="training"
    ).astype(DTYPE)

    df.loc[:, MODEL_FEATURES] = feature_block

    return df


############################################################
# END-TO-END WALK FORWARD TEST
############################################################

def test_walkforward_runs_end_to_end():

    df = build_synthetic_dataset()

    validator = WalkForwardValidator(
        model_trainer=dummy_trainer,
        window_size=120,
        step_size=30
    )

    metrics = validator.run(df)

    # ------------------------------------------------------
    # Core assertions
    # ------------------------------------------------------

    assert metrics["num_windows"] > 0
    assert metrics["final_equity"] > 0
    assert -5.0 <= metrics["avg_sharpe"] <= 5.0

    # ------------------------------------------------------
    # Stability checks
    # ------------------------------------------------------

    assert np.isfinite(metrics["avg_sharpe"])
    assert np.isfinite(metrics["final_equity"])
    assert metrics["profit_factor"] >= 0

    # Drawdown sanity
    assert metrics["max_drawdown"] <= 0
    assert metrics["max_drawdown"] >= -1.0

    # Turnover sanity
    assert metrics["avg_turnover"] >= 0
    assert metrics["avg_trades_per_window"] >= 1

    # ------------------------------------------------------
    # Ensure all expected metrics exist
    # ------------------------------------------------------

    required = {
        "avg_strategy_return",
        "avg_sharpe",
        "profit_factor",
        "max_drawdown",
        "return_volatility",
        "final_equity",
        "avg_turnover",
        "avg_win_rate",
        "avg_trades_per_window",
        "num_windows",
    }

    assert required.issubset(metrics.keys())


############################################################
# DETERMINISM TEST
############################################################

def test_walkforward_is_deterministic():

    df = build_synthetic_dataset()

    validator1 = WalkForwardValidator(
        model_trainer=dummy_trainer,
        window_size=120,
        step_size=30
    )

    validator2 = WalkForwardValidator(
        model_trainer=dummy_trainer,
        window_size=120,
        step_size=30
    )

    metrics1 = validator1.run(df.copy())
    metrics2 = validator2.run(df.copy())

    assert metrics1["avg_sharpe"] == metrics2["avg_sharpe"]
    assert metrics1["final_equity"] == metrics2["final_equity"]
    assert metrics1["profit_factor"] == metrics2["profit_factor"]