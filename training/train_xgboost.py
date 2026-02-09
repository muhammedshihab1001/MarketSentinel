import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from core.services import StockPriceFetcher
from core.services import NewsFetcher
from core.services import SentimentAnalyzer
from core.services import FeatureEngineer
from app.config.features import MODEL_FEATURES


MODEL_PATH = "models/xgboost_direction.pkl"
MIN_ACCURACY = 0.50


def load_training_data():
    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    fe = FeatureEngineer()

    # Price data
    price_df = fetcher.fetch(
        ticker="AAPL",
        start_date="2025-01-01",
        end_date="2025-12-31"
    )

    # Feature engineering (price)
    price_df = fe.add_returns(price_df)
    price_df = fe.add_volatility(price_df)
    price_df = fe.add_rsi(price_df)
    price_df = fe.add_macd(price_df)

    # News + sentiment
    news_df = news_fetcher.fetch("Apple stock", max_items=50)
    scored_df = sentiment_analyzer.analyze_dataframe(news_df)
    sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

    # Merge & final dataset
    merged_df = fe.merge_price_sentiment(price_df, sentiment_df)
    dataset = fe.create_ml_dataset(merged_df)

    return dataset


def train_model(df: pd.DataFrame):
    # FEATURES = [
    #     "return",
    #     "volatility",
    #     "rsi",
    #     "macd",
    #     "macd_signal",
    #     "avg_sentiment",
    #     "news_count",
    #     "sentiment_std",
    #     "return_lag1",
    #     "sentiment_lag1"
    # ]

    X = df[MODEL_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    if acc < MIN_ACCURACY:
        raise ValueError(
            f"Model accuracy {acc:.2f} below threshold {MIN_ACCURACY}"
        )

    return model


if __name__ == "__main__":
    df = load_training_data()
    model = train_model(df)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
