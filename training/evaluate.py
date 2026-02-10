import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    roc_auc_score,
    mean_absolute_error
)


# ---------------------------------------------------
# INTERNAL SAFETY
# ---------------------------------------------------

def _flatten(arr):
    """
    Prevent silent metric corruption from (N,1) arrays.
    """
    return np.asarray(arr).reshape(-1)


# ---------------------------------------------------
# XGBOOST / CLASSIFICATION
# ---------------------------------------------------

def evaluate_xgboost(y_true, y_pred, y_prob=None):
    """
    Institutional evaluation for classification models.

    Guarantees:
    - shape safety
    - ROC protection
    - backward compatibility
    """

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        )
    }

    # -----------------------------------------
    # ROC AUC
    # -----------------------------------------

    if y_prob is not None:

        y_prob = _flatten(y_prob)

        # Detect invalid probabilities
        if np.any(y_prob < 0) or np.any(y_prob > 1):
            raise RuntimeError(
                "Invalid probabilities detected in evaluation."
            )

        try:

            # Guard against single-class folds
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


# ---------------------------------------------------
# LSTM / REGRESSION
# ---------------------------------------------------

def evaluate_lstm(y_true, y_pred):
    """
    Institutional RMSE.

    Adds:
    - shape safety
    - NaN guard
    """

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    if np.isnan(y_pred).any():
        raise RuntimeError("NaNs detected in predictions.")

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return {
        "rmse": float(rmse)
    }


# ---------------------------------------------------
# PROPHET / FORECASTING
# ---------------------------------------------------

def evaluate_prophet(actual, predicted):
    """
    Institutional MAE with shape safety.
    """

    actual = _flatten(actual)
    predicted = _flatten(predicted)

    if len(actual) != len(predicted):
        raise RuntimeError(
            "Forecast length mismatch."
        )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    return {
        "mae": float(mae)
    }
