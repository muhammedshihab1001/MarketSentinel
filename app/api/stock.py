from app.models.model_loader import ModelLoader
from app.models.lstm_model import forecast_lstm
from app.models.prophet_model import forecast_prophet
from app.services.signal_engine import SignalEngine, fuse_decision

models = ModelLoader()
engine = SignalEngine()

@router.get("/full-prediction")
def full_prediction():
    # existing XGBoost inference result
    direction = "BUY"
    prob_up = 0.6

    recent_prices = [[p] for p in range(100, 160)]  # placeholder

    lstm_prices = forecast_lstm(
        models.lstm,
        models.lstm_scaler,
        recent_prices
    )

    prophet_out = forecast_prophet(models.prophet)

    final_signal = fuse_decision(
        direction,
        prob_up,
        lstm_prices,
        prophet_out["trend"]
    )

    return {
        "ticker": "AAPL",
        "signal": final_signal,
        "confidence": prob_up,
        "forecast": {
            "lstm_prices": lstm_prices,
            "prophet_trend": prophet_out
        }
    }
