import os
import datetime
import tempfile
import shutil
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier

from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import MODEL_FEATURES
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from core.monitoring.drift_detector import DriftDetector

from training.backtesting.walk_forward import WalkForwardValidator


MODEL_DIR = "artifacts/xgboost"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

SEED = 42
np.random.seed(SEED)


TRAINING_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"
]


# ---------------------------------------------------
# ATOMIC SAVE
# ---------------------------------------------------

def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=os.path.dirname(path)
    ) as tmp:

        joblib.dump(model, tmp.name)
        temp_name = tmp.name

    shutil.move(temp_name, path)

    if not os.path.exists(path):
        raise RuntimeError("Model write failed.")


# ---------------------------------------------------
# SAFE BASELINE (WRITE ONCE)
# ---------------------------------------------------

def ensure_baseline(df):

    detector = DriftDetector()

    if os.path.exists(detector.BASELINE_PATH):
        return

    detector.create_baseline(df[list(MODEL_FEATURES)])


# ---------------------------------------------------

def load_training_data():

    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    feature_store = FeatureStore()

    end_date = datetime.date.today().isoformat()

    datasets = []

    for ticker in TRAINING_TICKERS:

        price_df = fetcher.fetch(
            ticker=ticker,
            start_date="2018-01-01",
            end_date=end_date
        )

        news_df = news_fetcher.fetch(
            f"{ticker} stock",
            max_items=100
        )

        scored_df = sentiment_analyzer.analyze_dataframe(news_df)
        sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

        dataset = feature_store.get_features(
            price_df,
            sentiment_df,
            ticker=ticker
        )

        dataset["ticker"] = ticker
        datasets.append(dataset)

    df = pd.concat(datasets, ignore_index=True)
    df = df.sort_values(["date","ticker"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Training dataset empty.")

    return df, end_date


# ---------------------------------------------------
# DATE SPLIT (NO LEAKAGE)
# ---------------------------------------------------

def date_split(df):

    cutoff = df["date"].quantile(0.8)

    train = df[df["date"] <= cutoff]
    val = df[df["date"] > cutoff]

    return train, val


# ---------------------------------------------------

def train_full_model(df):

    train, val = date_split(df)

    X_train = train[list(MODEL_FEATURES)]
    y_train = train["target"]

    X_val = val[list(MODEL_FEATURES)]
    y_val = val["target"]

    model = XGBClassifier(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        early_stopping_rounds=75,
        random_state=SEED,
        tree_method="hist",
        predictor="cpu_predictor",
        n_jobs=1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "roc_auc": float(roc_auc_score(y_val, probs)),
        "logloss": float(log_loss(y_val, probs)),
    }

    importance = dict(
        zip(list(MODEL_FEATURES), model.feature_importances_.tolist())
    )

    return model, metrics, importance


# ---------------------------------------------------

if __name__ == "__main__":

    df, end_date = load_training_data()

    ensure_baseline(df)

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    wf = WalkForwardValidator(
        model_trainer=lambda d: train_full_model(d)[0],
        signal_generator=lambda m, d: np.where(
            m.predict_proba(d[list(MODEL_FEATURES)])[:,1] > 0.6,
            "BUY","HOLD"
        )
    )

    strategy_metrics = wf.run(df)

    model, metrics, importance = train_full_model(df)

    save_model_atomic(model, TEMP_MODEL_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics={**metrics, **strategy_metrics},
        features=list(MODEL_FEATURES),
        training_start="2018-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="model"
    )

    enriched_metadata = {
        **metadata,
        "feature_importance": importance,
        "training_rows": len(df),
        "feature_count": len(MODEL_FEATURES)
    }

    MetadataManager.save_metadata(
        enriched_metadata,
        TEMP_METADATA_PATH
    )

    version_dir = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print(f"XGBoost registered → {version_dir}")
