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
from core.artifacts.metadata_manager import MetadataManager


MODEL_DIR = "artifacts/xgboost"
MODEL_PATH = f"{MODEL_DIR}/model.pkl"
METADATA_PATH = f"{MODEL_DIR}/metadata.json"
MIN_ACCURACY = 0.50


# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------

def load_training_data():
    """
    Uses canonical feature pipeline.
    Guarantees training == inference features.
    """

    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()

    end_date = datetime.date.today().isoformat()

    price_df = fetcher.fetch(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date=end_date
    )

    news_df = news_fetcher.fetch("Apple stock", max_items=100)

    scored_df = sentiment_analyzer.analyze_dataframe(news_df)
    sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

    dataset = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df
    )

    return dataset, end_date


# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

def train_model(df: pd.DataFrame):

    X = df[MODEL_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
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

    return model, acc


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":

    df, end_date = load_training_data()
    model, acc = train_model(df)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Create metadata
    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics={"accuracy": float(acc)},
        features=MODEL_FEATURES,
        training_start="2018-01-01",
        training_end=end_date
    )

    MetadataManager.save_metadata(metadata, METADATA_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")
