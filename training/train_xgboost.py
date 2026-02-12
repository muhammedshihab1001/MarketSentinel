import os
import datetime
import tempfile
import shutil
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier

from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES, validate_feature_schema

from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from training.backtesting.walk_forward import WalkForwardValidator


MODEL_DIR = "artifacts/xgboost"
TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

SEED = 42
np.random.seed(SEED)

TRAINING_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
    "JPM","GS","BAC","AMD","AVGO"
]

MIN_TRAINING_ROWS = 2000
MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40


def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as tmp:
        joblib.dump(model, tmp.name)
        temp_name = tmp.name

    shutil.move(temp_name, path)

    if not os.path.exists(path):
        raise RuntimeError("Model write failed.")


def load_training_data():

    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    engineer = FeatureEngineer()

    end_date = datetime.date.today().isoformat()

    datasets = []

    for ticker in TRAINING_TICKERS:

        price_df = fetcher.fetch(
            ticker=ticker,
            start_date="2016-01-01",
            end_date=end_date
        )

        news_df = news_fetcher.fetch(
            f"{ticker} stock",
            max_items=100
        )

        scored_df = sentiment_analyzer.analyze_dataframe(news_df)
        sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

        dataset = engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=True
        )

        dataset["ticker"] = ticker
        datasets.append(dataset)

    df = pd.concat(datasets, ignore_index=True)

    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError(f"Training aborted — dataset too small ({len(df)} rows)")

    feature_block = validate_feature_schema(df)

    # KEEP close column for regime detector
    df = pd.concat(
        [
            df[["date", "ticker", "close", "target"]],
            feature_block
        ],
        axis=1
    )

    return df, end_date


def train_model(train_df):

    X = train_df[list(MODEL_FEATURES)]
    y = train_df["target"]

    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)

    model = XGBClassifier(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=SEED,
        tree_method="hist",
        n_jobs=1,
        scale_pos_weight=pos_weight
    )

    model.fit(X, y)

    return model


def generate_signals(model, test_df):

    probs = model.predict_proba(
        test_df[list(MODEL_FEATURES)]
    )[:, 1]

    return [
        "BUY" if p > 0.58 else "HOLD"
        for p in probs
    ]


def train_full_model(df):

    X = df[list(MODEL_FEATURES)]
    y = df["target"]

    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)

    model = XGBClassifier(
        n_estimators=900,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=SEED,
        tree_method="hist",
        n_jobs=1,
        scale_pos_weight=pos_weight
    )

    model.fit(X, y)

    importance = dict(
        zip(MODEL_FEATURES, model.feature_importances_.tolist())
    )

    return model, importance


if __name__ == "__main__":

    print("Institutional XGBoost Training")

    df, end_date = load_training_data()

    dataset_hash = MetadataManager.fingerprint_dataset(
        df[list(MODEL_FEATURES)]
    )

    wf = WalkForwardValidator(
        model_trainer=train_model,
        signal_generator=generate_signals
    )

    strategy_metrics = wf.run(df)

    if strategy_metrics["avg_sharpe"] < MIN_SHARPE:
        raise RuntimeError("XGBoost rejected — Sharpe too low")

    if strategy_metrics["max_drawdown"] < MAX_DRAWDOWN:
        raise RuntimeError("XGBoost rejected — drawdown too severe")

    model, importance = train_full_model(df)

    save_model_atomic(model, TEMP_MODEL_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics={**strategy_metrics},
        features=list(MODEL_FEATURES),
        training_start="2016-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="tabular"
    )

    metadata["feature_importance"] = importance
    metadata["training_rows"] = len(df)

    MetadataManager.save_metadata(
        metadata,
        TEMP_METADATA_PATH
    )

    version = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print(f"XGBoost registered → {version}")
