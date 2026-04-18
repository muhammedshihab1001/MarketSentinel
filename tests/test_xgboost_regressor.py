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
        np.random.randn(rows, features), columns=[f"f{i}" for i in range(features)]
    )

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
# FEATURE MISMATCH — FILLS WITH 0, NOT RAISES
# FIX: predict() fills missing columns with 0 (graceful inference flexibility).
# Only NaN VALUES raise — missing columns are a schema extension case.
############################################################


def test_predict_feature_mismatch_fills_gracefully():
    """
    FIX (item 20): predict() fills missing COLUMNS with 0 and logs a warning.
    It does NOT raise RuntimeError for missing columns — that would break
    inference when new features are added to schema before model is retrained.
    """

    X, y = generate_data()

    model = SafeXGBRegressor()
    model.fit(X, y)

    # Drop one column — predict should fill with 0, not crash
    X_missing_col = X.drop(columns=[X.columns[0]])

    # Should NOT raise — graceful fill
    preds = model.predict(X_missing_col)
    assert len(preds) == len(X_missing_col)
    assert np.all(np.isfinite(preds))


############################################################
# NAN INFERENCE GUARD
# FIX (item 20): NaN VALUES in input now raise RuntimeError.
# NaN = upstream pipeline bug that should surface loudly.
############################################################


def test_predict_nan_guard():
    """
    FIX (item 20): NaN in inference input raises RuntimeError.
    This surfaces upstream feature pipeline bugs instead of silently
    filling with 0 and producing wrong predictions.
    """

    X, y = generate_data()

    model = SafeXGBRegressor()
    model.fit(X, y)

    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan

    with pytest.raises(RuntimeError, match="NaN"):
        model.predict(X_nan)


############################################################
# TARGET VARIANCE GUARD
############################################################


def test_target_variance_guard():

    X = pd.DataFrame(np.random.randn(200, 10), columns=[f"f{i}" for i in range(10)])

    y = np.ones(200)  # zero variance target

    model = SafeXGBRegressor()

    with pytest.raises(RuntimeError):
        model.fit(X, y)


############################################################
# LOW DISPERSION WARNING (NON-FATAL)
############################################################


def test_low_dispersion_warning():

    X = pd.DataFrame(np.random.randn(300, 8), columns=[f"f{i}" for i in range(8)])

    y = np.random.randn(300)

    model = SafeXGBRegressor()
    model.fit(X, y)

    X_same = X.copy()
    X_same.iloc[:] = 0.5

    preds = model.predict(X_same)

    assert np.all(np.isfinite(preds))


############################################################
# FEATURE IMPORTANCE EXPORT  (item 19)
# FIX: export_feature_importance() now exists on SafeXGBRegressor
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

    # importance list should be sorted descending
    importances = [row["importance"] for row in info["feature_importance"]]
    assert importances == sorted(importances, reverse=True)

    # all features should be present or subset
    feat_names = {row["feature"] for row in info["feature_importance"]}
    assert feat_names.issubset(set(X.columns))

    # importances sum to ~1.0 (normalized)
    total = sum(importances)
    assert abs(total - 1.0) < 0.01
