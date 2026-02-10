import os
import datetime
import tempfile
import shutil
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    log_loss
)

from xgboost import XGBClassifier

from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

MODEL_DIR = "artifacts/xgboost"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

MIN_ACCURACY = 0.50


# ---------------------------------------------------
# SAFE MODEL WRITE
# ---------------------------------------------------

def save_model_atomic(model, path):
    """
    Prevents corrupted artifacts.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=os.path.dirname(path)
    ) as tmp:

        joblib.dump(model, tmp.name)
        temp_name = tmp.name

    shutil.move(temp_name, path)


# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------

def load_training_data():

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

    if dataset.empty:
        raise ValueError("Feature pipeline returned empty dataset")

    return dataset, end_date


# ---------------------------------------------------
# TRAIN MODEL
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
        n_estimators=1000,  # high ceiling
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    loss = log_loss(y_test, probs)

    print("\n✅ Accuracy:", acc)
    print("✅ ROC-AUC:", auc)
    print("✅ LogLoss:", loss)
    print(classification_report(y_test, preds))

    if acc < MIN_ACCURACY:
        raise ValueError(
            f"Model accuracy {acc:.2f} below threshold {MIN_ACCURACY}"
        )

    metrics = {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "logloss": float(loss),
        "training_rows": len(df),
        "feature_count": len(MODEL_FEATURES)
    }

    return model, metrics


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":

    print("\n🚀 Starting XGBoost training...\n")

    df, end_date = load_training_data()

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    model, metrics = train_model(df)

    save_model_atomic(model, TEMP_MODEL_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics=metrics,
        features=MODEL_FEATURES,
        training_start="2018-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash
    )

    MetadataManager.save_metadata(metadata, TEMP_METADATA_PATH)

    # ---------------------------------------------------
    # REGISTER MODEL
    # ---------------------------------------------------

    version_dir = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print("\n✅ XGBoost model registered successfully.")
    print(f"📦 Version directory: {version_dir}")
