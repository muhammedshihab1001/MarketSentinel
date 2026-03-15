import numpy as np
import pandas as pd
import pytest

from core.models.xgboost import SafeXGBRegressor


############################################################
# DATA GENERATOR
############################################################

def generate_data(rows=400, features=12):

    np.random.seed(42)

    X = pd.DataFrame(
        np.random.randn(rows, features),
        columns=[f"f{i}" for i in range(features)]
    )

    # Non-degenerate regression target
    y = np.random.randn(rows)

    return X, y


############################################################
# TRAINING TEST
############################################################

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

    # Ensure feature names stored correctly
    assert list(model.feature_names) == list(X.columns)


############################################################
# PREDICTION TEST
############################################################

def test_predict_valid():

    X, y = generate_data()

    model = SafeXGBRegressor()
    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds))
    assert np.std(preds) > 0


############################################################
# FEATURE MISMATCH TEST
############################################################

def test_predict_feature_mismatch():

    X, y = generate_data()

    model = SafeXGBRegressor()
    model.fit(X, y)

    X_bad = X.copy()
    X_bad = X_bad.drop(columns=[X_bad.columns[0]])

    with pytest.raises(RuntimeError):
        model.predict(X_bad)


############################################################
# NAN INFERENCE GUARD
############################################################

def test_predict_nan_guard():

    X, y = generate_data()

    model = SafeXGBRegressor()
    model.fit(X, y)

    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan

    with pytest.raises(RuntimeError):
        model.predict(X_nan)


############################################################
# TARGET VARIANCE GUARD
############################################################

def test_target_variance_guard():

    X = pd.DataFrame(np.random.randn(200, 10))
    y = np.ones(200)  # zero variance target

    model = SafeXGBRegressor()

    with pytest.raises(RuntimeError):
        model.fit(X, y)


############################################################
# LOW DISPERSION WARNING (NON-FATAL)
############################################################

def test_low_dispersion_warning():

    X = pd.DataFrame(np.random.randn(300, 8))
    y = np.random.randn(300)

    model = SafeXGBRegressor()
    model.fit(X, y)

    # Force identical inference features
    X_same = X.copy()
    X_same.iloc[:] = 0.5

    preds = model.predict(X_same)

    assert np.all(np.isfinite(preds))


############################################################
# FEATURE IMPORTANCE EXPORT
############################################################

def test_feature_importance_export():

    X, y = generate_data()

    model = SafeXGBRegressor()
    model.fit(X, y)

    info = model.export_feature_importance()

    assert "feature_importance" in info
    assert "booster_checksum" in info
    assert "feature_checksum" in info
    assert "train_rmse" in info