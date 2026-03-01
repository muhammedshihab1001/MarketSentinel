import numpy as np
import pandas as pd
import pytest

from core.models.xgboost import SafeXGBRegressor


def generate_data(rows=400, features=12):

    X = pd.DataFrame(
        np.random.randn(rows, features),
        columns=[f"f{i}" for i in range(features)]
    )

    # Create non-degenerate regression target
    y = np.random.randn(rows)

    return X, y


# ==========================================================
# TRAINING TEST
# ==========================================================

def test_training_and_checksums():

    X, y = generate_data()

    model = SafeXGBRegressor()
    model.fit(X, y)

    assert model.model is not None
    assert model.feature_checksum is not None
    assert model.param_checksum is not None
    assert model.booster_checksum is not None
    assert model.best_iteration is not None
    assert model.training_fingerprint is not None


# ==========================================================
# PREDICTION TEST
# ==========================================================

def test_predict_valid():

    X, y = generate_data()

    model = SafeXGBRegressor()
    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds))
    assert np.std(preds) > 0


# ==========================================================
# ENTROPY GUARD TEST
# ==========================================================

def test_entropy_guard_trigger():

    # Degenerate dataset (constant features + constant target)
    X = pd.DataFrame(np.ones((200, 10)))
    y = np.ones(200)

    model = SafeXGBRegressor()

    with pytest.raises(RuntimeError):
        model.fit(X, y)


# ==========================================================
# TARGET VARIANCE GUARD
# ==========================================================

def test_target_variance_guard():

    X = pd.DataFrame(np.random.randn(200, 10))
    y = np.ones(200)  # zero variance target

    model = SafeXGBRegressor()

    with pytest.raises(RuntimeError):
        model.fit(X, y)