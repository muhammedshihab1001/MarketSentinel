import os
import datetime
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES


MODEL_PATH = "artifacts/xgboost_direction.pkl"
MIN_ACCURACY = 0.50


def load_training_data():
    """
    Loads data using the canonical feature pipeline.
    Guarantees training == inference features.
    """

    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()

    end_date = datetime.date.today().isoformat()

    # -----------------------------
    # Fetch price data
    # -----------------------------
    price_df = fetcher.fetch(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date=end_date
    )

    # -----------------------------
    # Fetch sentiment
    # -----------------------------
    news_df = news_fetcher.fetch("Apple stock", max_items=100)

    scored_df = sentiment_analyzer.analyze_dataframe(news_df)
    sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

    # -----------------------------
    # Canonical feature pipeline
    # -----------------------------
    dataset = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df
    )

    return dataset


def train_model(df: pd.DataFrame):

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

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
