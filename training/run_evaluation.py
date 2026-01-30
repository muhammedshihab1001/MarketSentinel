from training.evaluate import (
    evaluate_xgboost,
    evaluate_lstm,
    evaluate_prophet,
    validate_metrics
)

# Example dummy values (replace with real outputs)
xgb_metrics = evaluate_xgboost([1,0,1,1], [1,0,0,1])
lstm_metrics = evaluate_lstm([180,182,181], [179,183,180])
prophet_metrics = evaluate_prophet(
    actual=[180,181,182],
    predicted=[179,182,183]
)

validate_metrics(xgb_metrics)
validate_metrics(lstm_metrics)
validate_metrics(prophet_metrics)

print("All model metrics passed quality gates")
