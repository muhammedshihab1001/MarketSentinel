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

def evaluate_xgboost(y_true, y_pred, y_prob):

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }


# ---------------------------------------------------
# LSTM
# ---------------------------------------------------

def evaluate_lstm(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "rmse": float(rmse)
    }


# ---------------------------------------------------
# PROPHET
# ---------------------------------------------------

def evaluate_prophet(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)

    return {
        "mae": float(mae)
    }
