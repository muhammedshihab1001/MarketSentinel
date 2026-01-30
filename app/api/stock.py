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

# Load models once at startup (important for performance)
models = ModelLoader()
signal_engine = SignalEngine()


@router.get("/full-prediction")
def full_prediction():
    """
    Full stock prediction endpoint:
    - Direction (XGBoost)
    - Short-term forecast (LSTM)
    - Long-term trend (Prophet)
    - Final BUY / SELL / HOLD decision
    """

    start_time = time.time()
    endpoint = "/full-prediction"

    try:
        # -----------------------------
        # Request metrics
        # -----------------------------
        REQUEST_COUNT.labels(endpoint=endpoint).inc()

        # -----------------------------
        # XGBoost (direction) – placeholder
        # -----------------------------
        # TODO: Replace with real XGBoost inference
        direction = "BUY"
        prob_up = 0.60

        # -----------------------------
        # LSTM forecast
        # -----------------------------
        # TODO: Replace placeholder with real recent prices
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
        # Final decision fusion
        # -----------------------------
        final_signal = fuse_decision(
            direction_signal=direction,
            prob_up=prob_up,
            lstm_prices=lstm_prices,
            prophet_trend=prophet_out["trend"]
        )

        # -----------------------------
        # Model metrics
        # -----------------------------
        PREDICTION_COUNT.labels(signal=final_signal).inc()
        AVG_CONFIDENCE.set(prob_up)

        response = {
            "ticker": "AAPL",
            "signal": final_signal,
            "confidence": prob_up,
            "models": {
                "direction_model": direction,
                "lstm_forecast": lstm_prices,
                "prophet_trend": prophet_out
            }
        }

        return response

    except Exception as e:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise e

    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )
