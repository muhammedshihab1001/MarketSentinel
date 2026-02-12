import os
import datetime
import tempfile
import shutil
import joblib
import pandas as pd
import numpy as np
import logging

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

#  NEW — signal safety thresholds
MIN_SENTIMENT_STD = 0.015
MIN_NEWS_PER_DAY = 1.2


########################################################
# ATOMIC SAVE
########################################################

def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as tmp:
        joblib.dump(model, tmp.name)
        temp_name = tmp.name

    shutil.move(temp_name, path)

    if not os.path.exists(path):
        raise RuntimeError("Model write failed.")


########################################################
# ELITE NEWS QUERY
########################################################

def build_news_query(ticker: str) -> str:
    """
    Institutional query design.
    Expands semantic surface area.
    """
    return (
        f"{ticker} OR {ticker} earnings OR {ticker} revenue "
        f"OR {ticker} guidance OR {ticker} forecast "
        f"OR {ticker} downgrade OR upgrade"
    )


########################################################
# SIGNAL DIAGNOSTICS (VERY IMPORTANT)
########################################################

def validate_sentiment_signal(sentiment_df, ticker):

    if sentiment_df.empty:
        raise RuntimeError(f"{ticker}: no sentiment data.")

    std = sentiment_df["avg_sentiment"].std()
    news_rate = sentiment_df["news_count"].mean()

    logger.info(
        "%s sentiment | std=%.4f | avg_news=%.2f",
        ticker,
        std,
        news_rate
    )

    if std < MIN_SENTIMENT_STD:
        raise RuntimeError(
            f"{ticker}: sentiment variance collapsed ({std:.5f})"
        )

    if news_rate < MIN_NEWS_PER_DAY:
        raise RuntimeError(
            f"{ticker}: insufficient news flow ({news_rate:.2f}/day)"
        )


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

    for ticker in TRAINING_TICKERS:

        logger.info("Building dataset for %s", ticker)

        try:

            price_df = fetcher.fetch(
                ticker=ticker,
                start_date="2016-01-01",
                end_date=end_date
            )

            if price_df is None or price_df.empty:
                continue

            ####################################################
            #  ELITE NEWS QUERY
            ####################################################

            query = build_news_query(ticker)

            news_df = news_fetcher.fetch(
                query,
                max_items=250
            )

            scored_df = sentiment_analyzer.analyze_dataframe(news_df)
            sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

            ####################################################
            #  SIGNAL VALIDATION
            ####################################################

            validate_sentiment_signal(sentiment_df, ticker)

            dataset = engineer.build_feature_pipeline(
                price_df,
                sentiment_df,
                training=True
            )

            if dataset is None or dataset.empty:
                continue

            if dataset["close"].isna().any():
                raise RuntimeError("NaN close detected.")

            dataset["ticker"] = ticker

            datasets.append(dataset)

        except Exception as e:

            logger.warning(
                "Ticker rejected: %s | reason=%s",
                ticker,
                str(e)
            )

            continue

    if not datasets:
        raise RuntimeError("All tickers rejected — no training dataset.")

    df = pd.concat(datasets, ignore_index=True)

    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError(
            f"Training aborted — dataset too small ({len(df)} rows)"
        )

    feature_block = validate_feature_schema(
        df.loc[:, MODEL_FEATURES]
    )

    final = pd.concat(
        [
            df[["date", "ticker", "close", "target"]].reset_index(drop=True),
            feature_block.reset_index(drop=True)
        ],
        axis=1
    )

    logger.info("FINAL DATASET SIZE: %s rows", len(final))

    return final, end_date


########################################################
# WALK FORWARD TRAINER
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

    return [
        "BUY" if p > 0.58 else "HOLD"
        for p in probs
    ]


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

    return model, importance


########################################################
# MAIN
########################################################

def main():

    print("Institutional XGBoost Training")

    df, end_date = load_training_data()

    drift = DriftDetector()

    dataset_hash = MetadataManager.fingerprint_dataset(
        df.loc[:, MODEL_FEATURES]
    )

    drift.create_baseline(
        df.loc[:, MODEL_FEATURES],
        dataset_hash=dataset_hash,
        allow_overwrite=True
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


if __name__ == "__main__":
    main()
