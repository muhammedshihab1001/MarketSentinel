import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def evaluate_xgboost(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred))
    }


def evaluate_lstm(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(rmse)
    }


def evaluate_prophet(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = np.mean(np.abs(actual - predicted))
    return {
        "mae": float(mae)
    }
