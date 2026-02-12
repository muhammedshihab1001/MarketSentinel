import os
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

from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse


logger = logging.getLogger("marketsentinel.training")

init_env()

MODEL_DIR = "artifacts/xgboost"
TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

SEED = 42
np.random.seed(SEED)

MIN_TRAINING_ROWS = 2000
MIN_SURVIVING_TICKERS = 5
MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40


########################################################
# ATOMIC SAVE
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
# NEWS QUERY
########################################################

def build_news_query(ticker: str):

    COMPANY_MAP = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "Nvidia",
        "AMZN": "Amazon",
        "GOOGL": "Google",
        "META": "Meta",
        "TSLA": "Tesla",
        "JPM": "JPMorgan",
        "GS": "Goldman Sachs",
        "BAC": "Bank of America",
        "AMD": "Advanced Micro Devices",
        "AVGO": "Broadcom",
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF"
    }

    name = COMPANY_MAP.get(ticker, ticker)

    return f'"{name}" OR "{ticker}" stock'


########################################################
# DATA LOADER — CLOCK + UNIVERSE GOVERNED
########################################################

def load_training_data(start_date, end_date):

    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    engineer = FeatureEngineer()

    universe = MarketUniverse.get_universe()

    logger.info(
        "Training window | %s -> %s | universe=%s",
        start_date,
        end_date,
        len(universe)
    )

    datasets = []
    surviving = []

    for ticker in universe:

        try:

            price_df = fetcher.fetch(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )

            if price_df is None or price_df.empty:
                continue

            ###########################################
            # NEWS RETRY
            ###########################################

            news_df = None

            for attempt in range(4):
                try:
                    news_df = news_fetcher.fetch(
                        build_news_query(ticker),
                        max_items=400
                    )

                    if news_df is not None and not news_df.empty:
                        break

                except Exception:
                    pass

                time.sleep(2 ** attempt)

            if news_df is None or news_df.empty:
                continue

            scored_df = sentiment_analyzer.analyze_dataframe(news_df)
            sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

            dataset = engineer.build_feature_pipeline(
                price_df,
                sentiment_df,
                training=True
            )

            dataset["ticker"] = ticker

            datasets.append(dataset)
            surviving.append(ticker)

        except Exception as e:
            logger.warning("Ticker rejected: %s | %s", ticker, str(e))

    ###################################################
    # COLLAPSE PROTECTION
    ###################################################

    if len(surviving) < MIN_SURVIVING_TICKERS:
        raise RuntimeError(
            "Universe collapse — too few tickers survived."
        )

    df = pd.concat(datasets, ignore_index=True)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError("Training aborted — dataset too small.")

    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    validate_feature_schema(df.loc[:, MODEL_FEATURES])

    return df, surviving


########################################################
# MAIN
########################################################

def main(start_date=None, end_date=None):

    logger.info("Institutional XGBoost Training")

    ###################################################
    # MODEL-SPECIFIC CLOCK
    ###################################################

    if not start_date or not end_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    df, surviving = load_training_data(start_date, end_date)

    ###################################################
    # GOVERNANCE HASHES
    ###################################################

    dataset_hash = MetadataManager.fingerprint_dataset(
        df[["ticker","date","target", *MODEL_FEATURES]]
    )

    universe_hash = MetadataManager.hash_list(surviving)

    ###################################################
    # DRIFT BASELINE
    ###################################################

    drift = DriftDetector()

    drift.create_baseline(
        df.loc[:, MODEL_FEATURES],
        dataset_hash=dataset_hash,
        allow_overwrite=False
    )

    ###################################################
    # WALK FORWARD
    ###################################################

    def trainer(d):
        model = build_xgboost_model(d["target"])
        model.fit(d.loc[:, MODEL_FEATURES], d["target"])
        return model

    def signal(model, test):
        probs = model.predict_proba(
            test.loc[:, MODEL_FEATURES]
        )[:, 1]

        return ["BUY" if p > 0.58 else "HOLD" for p in probs]

    wf = WalkForwardValidator(
        model_trainer=trainer,
        signal_generator=signal
    )

    strategy_metrics = wf.run(df)

    sharpe = strategy_metrics.get("avg_sharpe")

    if sharpe is None or not np.isfinite(sharpe):
        raise RuntimeError("Invalid Sharpe produced.")

    if sharpe < MIN_SHARPE:
        raise RuntimeError("XGBoost rejected — Sharpe too low")

    if strategy_metrics["max_drawdown"] < MAX_DRAWDOWN:
        raise RuntimeError("XGBoost rejected — drawdown too severe")

    ###################################################
    # FINAL TRAIN
    ###################################################

    model = build_final_xgboost_model(df["target"])
    model.fit(df.loc[:, MODEL_FEATURES], df["target"])

    save_model_atomic(model, TEMP_MODEL_PATH)

    ###################################################
    # METADATA (INSTITUTIONAL)
    ###################################################

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics={**strategy_metrics},
        features=list(MODEL_FEATURES),
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="training_manifest_v1",
        extra_fields={
            "training_universe": surviving,
            "universe_hash": universe_hash,
            "time_snapshot": MarketTime.snapshot_for("xgboost")
        }
    )

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
