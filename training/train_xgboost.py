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
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry

from training.backtesting.walk_forward import WalkForwardValidator


# ===================================================
# CONFIG — Institutional Gates
# ===================================================

MODEL_DIR = "artifacts/xgboost"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

MIN_ACCURACY = 0.50
MIN_SHARPE = 0.50
MAX_DRAWDOWN = -0.35
MIN_ALPHA = 0.0
MIN_CALMAR = 0.30   # 🔥 NEW


# ===================================================
# SAFE WRITE
# ===================================================

def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False,
                                     dir=os.path.dirname(path)) as tmp:

        joblib.dump(model, tmp.name)
        temp_name = tmp.name

    shutil.move(temp_name, path)


# ===================================================
# DATA
# ===================================================

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


# ===================================================
# FEATURE BASELINES (🔥 VERY IMPORTANT)
# ===================================================

def build_feature_baselines(df):

    baselines = {}

    numeric_cols = df[MODEL_FEATURES].select_dtypes(
        include="number"
    ).columns

    for col in numeric_cols:

        series = df[col].dropna()

        baselines[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max())
        }

    return baselines


# ===================================================
# WALK-FORWARD HELPERS
# ===================================================

def train_on_window(df):

    X = df[MODEL_FEATURES]
    y = df["target"]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X, y)

    return model


def generate_signals(model, df):

    probs = model.predict_proba(df[MODEL_FEATURES])[:, 1]

    signals = np.where(
        probs > 0.6, "BUY",
        np.where(probs < 0.4, "SELL", "HOLD")
    )

    return signals.tolist()


# ===================================================
# FINAL TRAIN
# ===================================================

def train_full_model(df):

    X = df[MODEL_FEATURES]
    y = df["target"]

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42
    )

    split = int(len(df) * 0.8)

    model.fit(
        X[:split],
        y[:split],
        eval_set=[(X[split:], y[split:])],
        verbose=False
    )

    preds = model.predict(X[split:])
    probs = model.predict_proba(X[split:])[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y[split:], preds)),
        "roc_auc": float(roc_auc_score(y[split:], probs)),
        "logloss": float(log_loss(y[split:], probs)),
    }

    return model, metrics


# ===================================================
# EXECUTION
# ===================================================

if __name__ == "__main__":

    print("\n🚀 Starting Institutional XGBoost Training...\n")

    df, end_date = load_training_data()

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    # 🔥 BUILD BASELINES
    feature_baselines = build_feature_baselines(df)

    # ---------------------------------------------------
    # WALK-FORWARD
    # ---------------------------------------------------

    wf = WalkForwardValidator(
        model_trainer=train_on_window,
        signal_generator=generate_signals
    )

    strategy_metrics = wf.run(df)

    # 🔥 CALMAR
    calmar = (
        strategy_metrics["avg_strategy_return"]
        / abs(strategy_metrics["max_drawdown"])
    )

    strategy_metrics["calmar_ratio"] = float(calmar)

    print("\nStrategy Metrics:")
    print(strategy_metrics)

    # ---------------------------------------------------
    # PROMOTION GATES
    # ---------------------------------------------------

    if strategy_metrics["avg_sharpe"] < MIN_SHARPE:
        raise RuntimeError("Model rejected: Sharpe too low")

    if strategy_metrics["max_drawdown"] < MAX_DRAWDOWN:
        raise RuntimeError("Model rejected: Drawdown too high")

    if strategy_metrics["avg_alpha"] < MIN_ALPHA:
        raise RuntimeError("Model rejected: No alpha")

    if calmar < MIN_CALMAR:
        raise RuntimeError("Model rejected: Calmar too low")

    print("\n✅ Strategy passed institutional gates.\n")

    # ---------------------------------------------------
    # TRAIN FINAL
    # ---------------------------------------------------

    model, metrics = train_full_model(df)

    save_model_atomic(model, TEMP_MODEL_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics={**metrics, **strategy_metrics},
        features=MODEL_FEATURES,
        training_start="2018-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash,
    )

    # 🔥 Attach baselines
    metadata["feature_baselines"] = feature_baselines

    MetadataManager.save_metadata(metadata, TEMP_METADATA_PATH)

    version_dir = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print("\n✅ XGBoost model registered.")
    print(f"📦 Version directory: {version_dir}")
