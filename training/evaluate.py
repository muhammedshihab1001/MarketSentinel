import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    roc_auc_score,
    mean_absolute_error
)


# ---------------------------------------------------
# XGBOOST
# ---------------------------------------------------

def evaluate_xgboost(y_true, y_pred, y_prob=None):
    """
    Institutional evaluation for classification models.

    Backward compatible:
        Old tests may call with only (y_true, y_pred)

    Safe behavior:
        If probabilities are missing -> ROC AUC is skipped.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

    # ROC requires probabilities
    if y_prob is not None:

        y_prob = np.asarray(y_prob)

        # Avoid crash if only one class present
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_prob)
            )
        except ValueError:
            metrics["roc_auc"] = 0.5  # neutral baseline

    else:
        metrics["roc_auc"] = None

    return metrics


# ---------------------------------------------------
# LSTM
# ---------------------------------------------------

def evaluate_lstm(y_true, y_pred):
    """
    Regression-safe RMSE.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return {
        "rmse": float(rmse)
    }


# ---------------------------------------------------
# PROPHET
# ---------------------------------------------------

def evaluate_prophet(actual, predicted):
    """
    MAE for forecasting models.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    return {
        "mae": float(mae)
    }
