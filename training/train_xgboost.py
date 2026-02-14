import os
import tempfile
import joblib
import pandas as pd
import numpy as np
import logging
import random
import gc

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_store import FeatureStore
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

MODEL_DIR = os.path.abspath("artifacts/xgboost")
TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

SEED = 42
MIN_TRAINING_ROWS = 2000
MIN_SURVIVING_RATIO = 0.35
MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40
MIN_MODEL_BYTES = 50_000


########################################################
# STRICT DETERMINISM
########################################################

def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    random.seed(SEED)
    np.random.seed(SEED)


########################################################
# FSYNC
########################################################

def _fsync_dir(directory):

    if os.name == "nt":
        return

    fd = os.open(directory, os.O_DIRECTORY)

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


########################################################
# ATOMIC MODEL SAVE
########################################################

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

    if os.path.getsize(path) < MIN_MODEL_BYTES:
        raise RuntimeError("Model artifact suspiciously small.")


########################################################
# DATA LOADER
########################################################

def load_training_data(start_date, end_date):

    market_data = MarketDataService()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    store = FeatureStore()

    universe = MarketUniverse.get_universe()

    datasets = []
    surviving = []
    failures = []

    for ticker in universe:

        try:

            price_df = market_data.get_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )

            sentiment_df = None

            try:

                news_df = news_fetcher.fetch(
                    f'"{ticker}" stock',
                    max_items=400
                )

                if news_df is not None and not news_df.empty:

                    scored = sentiment_analyzer.analyze_dataframe(news_df)
                    sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored)

            except Exception:
                logger.warning("News failed for %s — neutral fallback used.", ticker)

            dataset = store.get_features(
                price_df,
                sentiment_df,
                ticker=ticker,
                training=True
            )

            datasets.append(dataset)
            surviving.append(ticker)

        except Exception as e:

            failures.append(f"{ticker}: {str(e)}")
            logger.warning("Ticker rejected: %s | %s", ticker, str(e))

    if not datasets:
        raise RuntimeError("All tickers failed — training aborted.")

    survival_ratio = len(surviving) / max(len(universe), 1)

    logger.info(
        "Universe survival: %.2f%% (%s/%s)",
        survival_ratio * 100,
        len(surviving),
        len(universe)
    )

    if survival_ratio < MIN_SURVIVING_RATIO:
        raise RuntimeError(
            f"Universe collapse — survival ratio {survival_ratio:.2f}"
        )

    df = pd.concat(datasets, ignore_index=True)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError("Training aborted — dataset too small.")

    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    validate_feature_schema(df.loc[:, MODEL_FEATURES])

    if not np.isfinite(df.loc[:, MODEL_FEATURES].to_numpy()).all():
        raise RuntimeError("Non-finite values detected in training features.")

    if df["target"].nunique() < 2:
        raise RuntimeError("Target collapsed — cannot train classifier.")

    logger.info("Final training rows: %s", len(df))

    return df


########################################################
# MAIN
########################################################

def main(start_date=None, end_date=None):

    enforce_determinism()

    logger.info("Institutional XGBoost Training")

    if not start_date or not end_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    df = load_training_data(start_date, end_date)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    hash_df = df[["ticker", "date", "target", *MODEL_FEATURES]]

    dataset_hash = MetadataManager.fingerprint_dataset(hash_df)

    ###################################################
    # DRIFT
    ###################################################

    drift = DriftDetector()

    try:
        drift.create_baseline(
            hash_df,
            dataset_hash=dataset_hash,
            allow_overwrite=False
        )

    except RuntimeError as e:

        if "already exists" not in str(e):
            raise

        logger.info("Drift baseline already exists.")

    ###################################################
    # WALK FORWARD
    ###################################################

    def trainer(d):

        y = d["target"]

        model = build_xgboost_model(
            y,
            random_state=SEED,
            nthread=1,
            predictor="cpu_predictor",
            tree_method="hist"
        )

        model.fit(
            d.loc[:, MODEL_FEATURES],
            y
        )

        return model

    def signal(model, test):

        probs = model.predict_proba(
            test.loc[:, MODEL_FEATURES]
        )[:, 1]

        return [
            "BUY" if p > 0.58
            else "SELL" if p < 0.42
            else "HOLD"
            for p in probs
        ]

    wf = WalkForwardValidator(
        model_trainer=trainer,
        signal_generator=signal
    )

    strategy_metrics = wf.run(df)

    gc.collect()

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

    final_model = build_final_xgboost_model(
        df["target"],
        random_state=SEED,
        nthread=1,
        predictor="cpu_predictor",
        tree_method="hist"
    )

    final_model.fit(
        df.loc[:, MODEL_FEATURES],
        df["target"]
    )

    save_model_atomic(final_model, TEMP_MODEL_PATH)

    ###################################################
    # METADATA
    ###################################################

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics={**strategy_metrics},
        features=list(MODEL_FEATURES),
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        dataset_rows=len(df),
        metadata_type="training_manifest_v1",
        extra_fields={
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

    ModelRegistry.promote_to_production(
        MODEL_DIR,
        version
    )

    logger.info("XGBoost promoted → %s", version)

    return strategy_metrics


if __name__ == "__main__":
    main()
