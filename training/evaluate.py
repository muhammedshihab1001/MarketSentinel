import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    roc_auc_score,
    mean_absolute_error,
    balanced_accuracy_score
)


def _flatten(arr):
    return np.asarray(arr).reshape(-1)


###################################################
# XGBOOST / CLASSIFICATION
###################################################

def evaluate_xgboost(y_true, y_pred, y_prob=None):

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    if len(y_true) != len(y_pred):
        raise RuntimeError("Prediction length mismatch.")

    valid_labels = {0, 1}

    if not set(np.unique(y_true)).issubset(valid_labels):
        raise RuntimeError("Invalid classification labels detected.")

    if not set(np.unique(y_pred)).issubset(valid_labels):
        raise RuntimeError("Invalid prediction labels detected.")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        )
    }

    if y_prob is not None:

        y_prob = _flatten(y_prob)

        if len(y_prob) != len(y_true):
            raise RuntimeError("Probability length mismatch.")

        if np.any(y_prob < 0) or np.any(y_prob > 1):
            raise RuntimeError(
                "Invalid probabilities detected in evaluation."
            )

        if np.std(y_prob) < 1e-5:
            raise RuntimeError(
                "Probability distribution collapsed."
            )

        try:

            if len(np.unique(y_true)) < 2:
                metrics["roc_auc"] = 0.5
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob)
                )

        except Exception:
            metrics["roc_auc"] = 0.5

    else:
        metrics["roc_auc"] = None

    return metrics


###################################################
# LSTM / REGRESSION
###################################################

def evaluate_lstm(y_true, y_pred):

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    if len(y_true) != len(y_pred):
        raise RuntimeError("Prediction length mismatch.")

    if np.isnan(y_true).any():
        raise RuntimeError("NaNs detected in targets.")

    if np.isnan(y_pred).any():
        raise RuntimeError("NaNs detected in predictions.")

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return {
        "rmse": float(rmse)
    }


###################################################
# PROPHET / FORECASTING
###################################################

def evaluate_prophet(actual, predicted):

    actual = _flatten(actual)
    predicted = _flatten(predicted)

    if len(actual) != len(predicted):
        raise RuntimeError("Forecast length mismatch.")

    if np.isnan(actual).any():
        raise RuntimeError("NaNs detected in actual values.")

    if np.isnan(predicted).any():
        raise RuntimeError("NaNs detected in forecast.")

    mae = mean_absolute_error(
        actual,
        predicted
    )

    return {
        "mae": float(mae)
    }
