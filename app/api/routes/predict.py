from fastapi import APIRouter
import time

from app.models.model_loader import ModelLoader
from app.models.lstm_model import forecast_lstm
from app.models.prophet_model import forecast_prophet
from app.services.signal_engine import SignalEngine, fuse_decision
from app.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
    PREDICTION_COUNT,
    AVG_CONFIDENCE
)

router = APIRouter()

models = ModelLoader()
engine = SignalEngine()

@router.get("/full-prediction")
def full_prediction():
    start_time = time.time()

    REQUEST_COUNT.labels(endpoint="/full-prediction").inc()

    try:
        # -----------------------------
        # XGBoost result (placeholder)
        # -----------------------------
        prediction = 1        # 1 = UP, 0 = DOWN
        prob_up = 0.6
        avg_sentiment = 0.25
        volatility = 0.03

        # -----------------------------
        # Base signal
        # -----------------------------
        base_signal = engine.generate_signal(
            prediction=prediction,
            prob_up=prob_up,
            avg_sentiment=avg_sentiment,
            volatility=volatility
        )

        # -----------------------------
        # LSTM forecast
        # -----------------------------
        recent_prices = [[p] for p in range(100, 160)]
        lstm_prices = forecast_lstm(
            models.lstm,
            models.lstm_scaler,
            recent_prices
        )

        # -----------------------------
        # Prophet forecast
        # -----------------------------
        prophet_out = forecast_prophet(models.prophet)

        # -----------------------------
        # Final fused decision
        # -----------------------------
        final_signal = fuse_decision(
            direction_signal=base_signal,
            prob_up=prob_up,
            lstm_prices=lstm_prices,
            prophet_trend=prophet_out["trend"]
        )

        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/full-prediction").observe(latency)
        PREDICTION_COUNT.labels(ticker="AAPL").inc()
        AVG_CONFIDENCE.labels(ticker="AAPL").set(prob_up)


        return {
            "ticker": "AAPL",
            "signal": final_signal,
            "confidence": prob_up,
            "details": {
                "base_signal": base_signal,
                "prophet_trend": prophet_out["trend"]
            }
        }

    except Exception as e:
        ERROR_COUNT.labels(endpoint="/full-prediction").inc()
        raise e
