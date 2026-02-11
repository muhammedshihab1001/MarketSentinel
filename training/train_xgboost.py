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

MIN_SHARPE = 0.50
MAX_DRAWDOWN = -0.35
MIN_ALPHA = 0.0
MIN_CALMAR = 0.30

SEED = 42

TRAINING_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA"
]

np.random.seed(SEED)


# ---------------------------------------------------
# SAFE WRITE
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


# ---------------------------------------------------
# DATA LOADING (INSTITUTIONAL FIX)
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

        # CRITICAL — USE FEATURE STORE
        dataset = feature_store.get_features(
            price_df,
            sentiment_df,
            ticker=ticker
        )

        dataset["ticker"] = ticker

        datasets.append(dataset)

    df = pd.concat(datasets, ignore_index=True)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Training dataset is empty.")

    years = pd.to_datetime(df["date"]).dt.year.nunique()

    if years < 4:
        raise RuntimeError("Dataset lacks regime diversity.")

    up_ratio = df["target"].mean()

    if not 0.35 < up_ratio < 0.65:
        raise RuntimeError(f"Target imbalance detected: {round(up_ratio,3)}")

    print(f"Training rows: {len(df)}")
    print(f"Positive class ratio: {round(up_ratio,3)}")

    return df, end_date


# ---------------------------------------------------

def build_feature_baselines(df):

    baselines = {}

    for col in MODEL_FEATURES:

        series = df[col].dropna()

        baselines[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max())
        }

    return baselines


# ---------------------------------------------------

def compute_sample_weights(df):

    vol = df["volatility"].fillna(df["volatility"].median())

    weights = 1 / (vol + 1e-6)
    weights = weights / weights.mean()

    return weights


# ---------------------------------------------------

def train_on_window(df):

    X = df[list(MODEL_FEATURES)]
    y = df["target"]

    weights = compute_sample_weights(df)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        tree_method="hist",
        max_bin=256,
        n_jobs=1
    )

    model.fit(X, y, sample_weight=weights)

    return model


def generate_signals(model, df):

    probs = model.predict_proba(df[list(MODEL_FEATURES)])[:, 1]

    signals = np.where(
        probs > 0.6, "BUY",
        np.where(probs < 0.4, "SELL", "HOLD")
    )

    return signals.tolist()


# ---------------------------------------------------

def train_full_model(df):

    X = df[list(MODEL_FEATURES)]
    y = df["target"]

    weights = compute_sample_weights(df)

    split = int(len(df) * 0.8)

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
        max_bin=256,
        n_jobs=1
    )

    model.fit(
        X.iloc[:split],
        y.iloc[:split],
        sample_weight=weights.iloc[:split],
        eval_set=[(X.iloc[split:], y.iloc[split:])],
        verbose=False
    )

    preds = model.predict(X.iloc[split:])
    probs = model.predict_proba(X.iloc[split:])[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y.iloc[split:], preds)),
        "roc_auc": float(roc_auc_score(y.iloc[split:], probs)),
        "logloss": float(log_loss(y.iloc[split:], probs)),
    }

    importance = dict(
        zip(list(MODEL_FEATURES), model.feature_importances_.tolist())
    )

    return model, metrics, importance


# ---------------------------------------------------

if __name__ == "__main__":

    print("Starting Institutional XGBoost Training")

    df, end_date = load_training_data()

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    feature_baselines = build_feature_baselines(df)

    DriftDetector().create_baseline(df[list(MODEL_FEATURES)])

    wf = WalkForwardValidator(
        model_trainer=train_on_window,
        signal_generator=generate_signals
    )

    strategy_metrics = wf.run(df)

    drawdown = abs(strategy_metrics["max_drawdown"]) or 1e-6
    calmar = strategy_metrics["avg_strategy_return"] / drawdown

    strategy_metrics["calmar_ratio"] = float(calmar)

    if strategy_metrics["avg_sharpe"] < MIN_SHARPE:
        raise RuntimeError("Model rejected: Sharpe too low")

    if strategy_metrics["max_drawdown"] < MAX_DRAWDOWN:
        raise RuntimeError("Model rejected: Drawdown too high")

    if strategy_metrics["avg_alpha"] < MIN_ALPHA:
        raise RuntimeError("Model rejected: No alpha")

    if calmar < MIN_CALMAR:
        raise RuntimeError("Model rejected: Calmar too low")

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

    metadata["feature_baselines"] = feature_baselines
    metadata["feature_importance"] = importance
    metadata["training_rows"] = len(df)
    metadata["feature_count"] = len(MODEL_FEATURES)

    MetadataManager.save_metadata(
        metadata,
        TEMP_METADATA_PATH
    )

    version_dir = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print("XGBoost model registered.")
    print(f"Version directory: {version_dir}")
