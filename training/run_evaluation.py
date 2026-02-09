from training.evaluate import (
    evaluate_xgboost,
    evaluate_lstm,
    evaluate_prophet
)

from training.predict import (
    get_xgb_predictions,
    get_lstm_predictions,
    get_prophet_predictions
)


def main():
    # Get predictions
    y_true_xgb, y_pred_xgb = get_xgb_predictions()
    y_true_lstm, y_pred_lstm = get_lstm_predictions()
    y_true_prophet, y_pred_prophet = get_prophet_predictions()

    # Evaluate models
    xgb_metrics = evaluate_xgboost(y_true_xgb, y_pred_xgb)
    lstm_metrics = evaluate_lstm(y_true_lstm, y_pred_lstm)
    prophet_metrics = evaluate_prophet(y_true_prophet, y_pred_prophet)

    # Quality gates
    assert xgb_metrics["accuracy"] >= 0.55, "XGBoost accuracy below threshold"
    assert lstm_metrics["rmse"] <= 5.0, "LSTM RMSE too high"
    assert prophet_metrics["mae"] <= 5.0, "Prophet MAE too high"

    print("All model quality checks passed")
    print("Metrics:", xgb_metrics, lstm_metrics, prophet_metrics)


if __name__ == "__main__":
    main()
