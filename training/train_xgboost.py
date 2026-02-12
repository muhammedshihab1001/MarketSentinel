import os
import datetime
import tempfile
import joblib
import pandas as pd
import numpy as np
import logging
import time

from core.config.env_loader import init_env
from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES, validate_feature_schema

from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from core.monitoring.drift_detector import DriftDetector

from training.backtesting.walk_forward import WalkForwardValidator

from models.xgboost_model import (
    build_xgboost_model,
    build_final_xgboost_model
)

logger = logging.getLogger("marketsentinel.training")

init_env()

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
MIN_SURVIVING_TICKERS = 4
MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40

MIN_SENTIMENT_STD = 0.01
MIN_NEWS_PER_DAY = 0.6


########################################################
# ATOMIC SAVE (FSYNC SAFE)
########################################################

def _fsync_dir(directory):

    if os.name == "nt":
        return

    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_model_atomic(model, path):

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        joblib.dump(model, tmp.name)
        temp_name = tmp.name

    os.replace(temp_name, path)

    _fsync_dir(directory)

    if not os.path.exists(path):
        raise RuntimeError("Model write failed.")


########################################################
# SMART QUERY
########################################################

def build_news_query(ticker: str) -> str:
    return f"{ticker} earnings revenue guidance forecast upgrade downgrade"


########################################################
# SENTIMENT VALIDATION
########################################################

def validate_sentiment_signal(sentiment_df, ticker):

    if sentiment_df.empty:
        logger.warning("%s has no sentiment — skipping ticker.", ticker)
        return False

    std = sentiment_df["avg_sentiment"].std()
    news_rate = sentiment_df["news_count"].mean()

    if std < MIN_SENTIMENT_STD:
        sentiment_df["avg_sentiment"] *= 0.25

    if news_rate < MIN_NEWS_PER_DAY:
        sentiment_df["avg_sentiment"] *= 0.5

    return True


########################################################
# DATA LOADER
########################################################

def load_training_data():

    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    engineer = FeatureEngineer()

    end_date = datetime.date.today().isoformat()

    datasets = []
    surviving_tickers = []

    for ticker in TRAINING_TICKERS:

        try:

            price_df = fetcher.fetch(
                ticker=ticker,
                start_date="2016-01-01",
                end_date=end_date
            )

            if price_df is None or price_df.empty:
                continue

            for attempt in range(3):
                try:
                    news_df = news_fetcher.fetch(
                        build_news_query(ticker),
                        max_items=200
                    )
                    break
                except Exception:
                    time.sleep(2 ** attempt)
            else:
                continue

            scored_df = sentiment_analyzer.analyze_dataframe(news_df)
            sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

            if not validate_sentiment_signal(sentiment_df, ticker):
                continue

            dataset = engineer.build_feature_pipeline(
                price_df,
                sentiment_df,
                training=True
            )

            if dataset is None or dataset.empty:
                continue

            dataset["ticker"] = ticker

            datasets.append(dataset)
            surviving_tickers.append(ticker)

        except Exception as e:
            logger.warning("Ticker rejected: %s | %s", ticker, str(e))

    if len(surviving_tickers) < MIN_SURVIVING_TICKERS:
        raise RuntimeError("Too few tickers survived.")

    df = pd.concat(datasets, ignore_index=True)

    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError("Training aborted — dataset too small.")

    validate_feature_schema(df.loc[:, MODEL_FEATURES])

    return df, end_date


########################################################
# WALK FORWARD
########################################################

def train_model(train_df):

    X = train_df.loc[:, MODEL_FEATURES]
    y = train_df["target"]

    model = build_xgboost_model(y)
    model.fit(X, y)

    return model


def generate_signals(model, test_df):

    probs = model.predict_proba(
        test_df.loc[:, MODEL_FEATURES]
    )[:, 1]

    return ["BUY" if p > 0.58 else "HOLD" for p in probs]


########################################################
# FINAL TRAIN
########################################################

def train_full_model(df):

    X = df.loc[:, MODEL_FEATURES]
    y = df["target"]

    model = build_final_xgboost_model(y)
    model.fit(X, y)

    importance = dict(
        zip(MODEL_FEATURES, model.feature_importances_.tolist())
    )

    if len(importance) != len(MODEL_FEATURES):
        raise RuntimeError("Feature importance mismatch.")

    return model, importance


########################################################
# MAIN
########################################################

def main():

    logger.info("Institutional XGBoost Training")

    df, end_date = load_training_data()

    dataset_hash = MetadataManager.fingerprint_dataset(
        df[["ticker","date","target", *MODEL_FEATURES]]
    )

    drift = DriftDetector()

    drift.create_baseline(
        df.loc[:, MODEL_FEATURES],
        dataset_hash=dataset_hash,
        allow_overwrite=False
    )

    wf = WalkForwardValidator(
        model_trainer=train_model,
        signal_generator=generate_signals
    )

    strategy_metrics = wf.run(df)

    sharpe = strategy_metrics.get("avg_sharpe")

    if sharpe is None or not np.isfinite(sharpe):
        raise RuntimeError("Invalid Sharpe produced.")

    if sharpe < MIN_SHARPE:
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
        metadata_type="training_manifest_v1"
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

    logger.info("XGBoost registered → %s", version)

    return strategy_metrics


if __name__ == "__main__":
    main()
