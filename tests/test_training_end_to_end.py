import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from training.train_xgboost import trainer
from training.backtesting.walk_forward import WalkForwardValidator
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    DTYPE
)


############################################################
# SYNTHETIC DATASET BUILDER
############################################################

def build_training_dataset():

    rng = np.random.default_rng(42)

    dates = pd.date_range("2025-01-01", periods=400)
    tickers = [f"T{i}" for i in range(12)]

    rows = []

    for d in dates:
        for t in tickers:

            row = {
                "date": d,
                "ticker": t,

                # slightly trending synthetic price
                "close": 100 + rng.normal(0, 1) + (d.day * 0.02),

                "volatility": abs(rng.normal()) + 0.2,

                "regime": "SIDEWAYS",
                "market_regime": "SIDEWAYS",
                "regime_multiplier": 1.0
            }

            for f in MODEL_FEATURES:
                row[f] = rng.normal()

            rows.append(row)

    df = pd.DataFrame(rows)

    # enforce schema compliance
    feature_block = validate_feature_schema(
        df.loc[:, MODEL_FEATURES],
        mode="training"
    ).astype(DTYPE)

    df.loc[:, MODEL_FEATURES] = feature_block

    return df


############################################################
# TRAINING PIPELINE TEST
############################################################

def test_training_pipeline_runs():

    df = build_training_dataset()

    model = trainer(df)

    assert model is not None
    assert hasattr(model, "predict")


############################################################
# WALK-FORWARD WITH REAL TRAINER
############################################################

def test_training_walkforward():

    df = build_training_dataset()

    validator = WalkForwardValidator(
        model_trainer=trainer,
        window_size=150,
        step_size=40
    )

    metrics = validator.run(df)

    assert metrics["num_windows"] > 0
    assert metrics["final_equity"] > 0
    assert np.isfinite(metrics["avg_sharpe"])


############################################################
# ARTIFACT EXPORT TEST
############################################################

def test_model_artifact_export():

    from training.train_xgboost import export_artifacts

    df = build_training_dataset()

    model = trainer(df)

    metrics = {
        "avg_sharpe": 0.5,
        "avg_strategy_return": 0.01
    }

    dataset_hash = "dummy_hash"

    tmpdir = tempfile.mkdtemp()

    # redirect artifact directory
    os.environ["XGB_REGISTRY_DIR"] = tmpdir

    try:

        version = export_artifacts(
            model=model,
            metrics=metrics,
            dataset_hash=dataset_hash,
            dataset_rows=len(df),
            start_date="2025-01-01",
            end_date="2025-12-31",
            training_df=df,
            promote_baseline=False,
            create_baseline=False
        )

        assert version is not None

    finally:

        shutil.rmtree(tmpdir, ignore_errors=True)


############################################################
# DETERMINISM CHECK
############################################################

def test_training_deterministic():

    df = build_training_dataset()

    model1 = trainer(df.copy())
    model2 = trainer(df.copy())

    X = df.loc[:, MODEL_FEATURES]

    X = validate_feature_schema(
        X,
        mode="inference"
    ).astype(DTYPE)

    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    assert np.allclose(pred1, pred2)