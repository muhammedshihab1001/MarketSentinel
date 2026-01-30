import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def evaluate_xgboost(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred)
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


def validate_metrics(metrics):
    errors = []

    if "accuracy" in metrics and metrics["accuracy"] < 0.52:
        errors.append(f"XGBoost accuracy too low: {metrics['accuracy']}")

    if "rmse" in metrics and metrics["rmse"] > metrics.get("price_threshold", 10):
        errors.append(f"LSTM RMSE too high: {metrics['rmse']}")

    if "mae" in metrics and metrics["mae"] > metrics.get("price_threshold", 10):
        errors.append(f"Prophet MAE too high: {metrics['mae']}")

    if errors:
        raise ValueError(" | ".join(errors))

    return True
