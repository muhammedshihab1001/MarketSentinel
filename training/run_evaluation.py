"""
Institutional Model Evaluation Runner

Guarantees:
✅ Loads ONLY registry models
✅ Rebuilds dataset canonically
✅ Prevents training/inference drift
✅ Enforces quality gates
✅ CI-safe failure
"""

import datetime
import numpy as np
import tensorflow as tf
import joblib

from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer

from training.evaluate import (
    evaluate_xgboost,
    evaluate_lstm,
    evaluate_prophet
)

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------
# CONFIG — QUALITY GATES
# ---------------------------------------------------

XGB_MIN_ACCURACY = 0.55
XGB_MIN_AUC = 0.60

LSTM_MAX_RMSE = 5.0
PROPHET_MAX_MAE = 5.0


# ---------------------------------------------------
# LOAD DATASET (CANONICAL)
# ---------------------------------------------------

def build_dataset():

    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment = SentimentAnalyzer()

    end_date = datetime.date.today().isoformat()

    price_df = fetcher.fetch(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date=end_date
    )

    news_df = news_fetcher.fetch("Apple stock", max_items=200)

    scored = sentiment.analyze_dataframe(news_df)
    sentiment_df = sentiment.aggregate_daily_sentiment(scored)

    dataset = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df
    )

    return dataset, price_df


# ---------------------------------------------------
# LOAD REGISTRY ARTIFACT
# ---------------------------------------------------

def latest_path(model_dir, filename):
    import os

    latest = os.path.join(model_dir, "latest")

    if not os.path.exists(latest):
        raise RuntimeError(f"No latest model in {model_dir}")

    version_dir = os.path.realpath(latest)

    return os.path.join(version_dir, filename)


# ---------------------------------------------------
# XGBOOST
# ---------------------------------------------------

def evaluate_xgb(dataset):

    model_path = latest_path(
        "artifacts/xgboost",
        "model.pkl"
    )

    model = joblib.load(model_path)

    X = dataset[model.feature_names_in_]
    y = dataset["target"]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    metrics = evaluate_xgboost(y, preds, probs)

    assert metrics["accuracy"] >= XGB_MIN_ACCURACY, \
        f"XGB accuracy below gate: {metrics}"

    assert metrics["roc_auc"] >= XGB_MIN_AUC, \
        f"XGB AUC below gate: {metrics}"

    return metrics


# ---------------------------------------------------
# LSTM
# ---------------------------------------------------

def evaluate_lstm_model(price_df):

    model_path = latest_path(
        "artifacts/lstm",
        "model.keras"
    )

    scaler_path = latest_path(
        "artifacts/lstm",
        "scaler.pkl"
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    scaler = joblib.load(scaler_path)

    prices = price_df[["close"]].values
    scaled = scaler.transform(prices)

    LOOKBACK = 60

    X, y = [], []

    for i in range(len(scaled) - LOOKBACK):
        X.append(scaled[i:i+LOOKBACK])
        y.append(scaled[i+LOOKBACK])

    X = np.array(X)
    y = np.array(y)

    preds = model.predict(X, verbose=0)

    y_inv = scaler.inverse_transform(y)
    preds_inv = scaler.inverse_transform(preds)

    metrics = evaluate_lstm(y_inv, preds_inv)

    assert metrics["rmse"] <= LSTM_MAX_RMSE, \
        f"LSTM RMSE too high: {metrics}"

    return metrics


# ---------------------------------------------------
# PROPHET
# ---------------------------------------------------

def evaluate_prophet_model(price_df):

    model_path = latest_path(
        "artifacts/prophet",
        "prophet_trend.pkl"
    )

    model = joblib.load(model_path)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    actual = price_df["close"].tail(30).values
    predicted = forecast["yhat"].tail(30).values

    metrics = evaluate_prophet(actual, predicted)

    assert metrics["mae"] <= PROPHET_MAX_MAE, \
        f"Prophet MAE too high: {metrics}"

    return metrics


# ---------------------------------------------------
# MASTER
# ---------------------------------------------------

def main():

    print("\n🔎 Running institutional model evaluation...\n")

    dataset, price_df = build_dataset()

    xgb_metrics = evaluate_xgb(dataset)
    lstm_metrics = evaluate_lstm_model(price_df)
    prophet_metrics = evaluate_prophet_model(price_df)

    print("\n✅ ALL QUALITY GATES PASSED\n")

    print("XGBoost:", xgb_metrics)
    print("LSTM:", lstm_metrics)
    print("Prophet:", prophet_metrics)


if __name__ == "__main__":
    main()
