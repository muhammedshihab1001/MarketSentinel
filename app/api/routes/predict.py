import joblib
import pandas as pd

from fastapi import APIRouter, HTTPException
from app.services.data_fetcher import StockPriceFetcher
from app.services.feature_engineering import FeatureEngineer
from app.services.signal_engine import SignalEngine
from app.config.features import MODEL_FEATURES

MODEL_PATH = "models/xgboost_direction.pkl"

router = APIRouter()

fetcher = StockPriceFetcher()
fe = FeatureEngineer()
signal_engine = SignalEngine()

@router.get("/")
def predict_stock():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        raise HTTPException(status_code=500, detail="Model not available")

    # Fetch recent data
    price_df = fetcher.fetch(
        ticker="AAPL",
        start_date="2024-01-01",
        end_date="2026-01-01"
    )

    # Feature engineering (same as training)
    price_df = fe.add_returns(price_df)
    price_df = fe.add_volatility(price_df)
    price_df = fe.add_rsi(price_df)
    price_df = fe.add_macd(price_df)

    # Sentiment placeholders (real-time safe)
    price_df["avg_sentiment"] = 0.0
    price_df["news_count"] = 0
    price_df["sentiment_std"] = 0.0

    # Lag features
    price_df["return_lag1"] = price_df["return"].shift(1)
    price_df["sentiment_lag1"] = price_df["avg_sentiment"].shift(1)

    price_df = price_df.dropna().reset_index(drop=True)
    latest = price_df.iloc[-1:]

    X = latest[MODEL_FEATURES]

    prediction = int(model.predict(X)[0])
    prob_up = float(model.predict_proba(X)[0][1])

    signal = signal_engine.generate_signal(
        prediction=prediction,
        prob_up=prob_up,
        avg_sentiment=0.0,   # sentiment integration comes next
        volatility=float(latest["volatility"].values[0])
    )

    return {
        "ticker": "AAPL",
        "prediction": "UP" if prediction == 1 else "DOWN",
        "probability_up": round(prob_up, 3),
        "signal": signal
    }
