from training.evaluate import (
    evaluate_xgboost,
    evaluate_lstm,
    evaluate_prophet
)


def test_xgboost_metrics():
    y_true = [1, 0, 1]
    y_pred = [1, 0, 1]
    metrics = evaluate_xgboost(y_true, y_pred)
    assert "accuracy" in metrics


def test_lstm_metrics():
    y_true = [10, 12, 14]
    y_pred = [11, 13, 15]
    metrics = evaluate_lstm(y_true, y_pred)
    assert "rmse" in metrics


def test_prophet_metrics():
    y_true = [20, 22, 24]
    y_pred = [21, 23, 25]
    metrics = evaluate_prophet(y_true, y_pred)
    assert "mae" in metrics
