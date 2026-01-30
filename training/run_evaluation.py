from training.evaluate import (
    evaluate_xgboost,
    evaluate_lstm,
    evaluate_prophet
)

def main():
    xgb_metrics = evaluate_xgboost()
    lstm_metrics = evaluate_lstm()
    prophet_metrics = evaluate_prophet()

    # Quality gates
    assert xgb_metrics["accuracy"] >= 0.55
    assert lstm_metrics["mae"] <= 5.0
    assert prophet_metrics["mae"] <= 6.0

    print("All model quality checks passed")

if __name__ == "__main__":
    main()
