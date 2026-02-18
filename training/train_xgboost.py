import os
import tempfile
import shutil
import time
import joblib
import pandas as pd
import numpy as np
import logging
import random
import gc

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature
)

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


logger = logging.getLogger(__name__)

MODEL_DIR = os.path.abspath("artifacts/xgboost")

TEMP_DIR = tempfile.mkdtemp(prefix="xgb_train_")
TEMP_MODEL_PATH = os.path.join(TEMP_DIR, "model.pkl")
TEMP_METADATA_PATH = os.path.join(TEMP_DIR, "metadata.json")

SEED = 42

MIN_TRAINING_ROWS = 1200
MIN_SURVIVING_RATIO = 0.20
MIN_MODEL_BYTES = 50_000

PROMOTE_FLAG = os.getenv("ALLOW_MODEL_PROMOTION", "false").lower() == "true"
ALLOW_BASELINE_FLAG = os.getenv("ALLOW_DRIFT_BASELINE", "false").lower() == "true"

BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.58"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "0.42"))


############################################################
# DETERMINISM
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# ATOMIC SAVE
############################################################

def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = f"{path}.tmp"

    joblib.dump(model, tmp_path)
    os.replace(tmp_path, path)

    size = os.path.getsize(path)

    if size < MIN_MODEL_BYTES:
        raise RuntimeError("Model artifact suspiciously small.")

    joblib.load(path)

    return size


############################################################
# METRIC SANITIZER
############################################################

def sanitize_metrics(metrics: dict) -> dict:

    numeric = {}

    for k, v in metrics.items():

        if isinstance(v, (int, float, np.integer, np.floating)):
            numeric[k] = float(v)
        else:
            logger.info(
                "Dropping non-numeric metric → %s",
                k
            )

    if not numeric:
        raise RuntimeError("No numeric metrics available.")

    return numeric


############################################################
# CROSS-SECTIONAL NORMALIZATION
############################################################

def cross_sectional_normalize(df):

    df = df.sort_values(["date", "ticker"]).copy()

    counts = df.groupby("date")["ticker"].transform("count")

    if (counts < 2).any():
        logger.warning(
            "Dropping single-asset dates before normalization."
        )
        df = df[counts >= 2]

    grouped = df.groupby("date")

    for col in MODEL_FEATURES:

        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std")

        std = std.fillna(1.0).clip(lower=1e-6)

        df[col] = (df[col] - mean) / std

    return df.reset_index(drop=True)


############################################################
# DATA LOADER — PRICE ONLY MODE
############################################################

def load_training_data(start_date, end_date):

    logger.info("Loading institutional training dataset...")

    market_data = MarketDataService()
    store = FeatureStore()

    universe = MarketUniverse.get_universe()

    datasets = []
    surviving = []

    logger.warning(
        "TRAINING MODE: Sentiment disabled — price-only research."
    )

    for ticker in universe:

        try:

            price_df = market_data.get_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )

            dataset = store.get_features(
                price_df,
                sentiment_df=None,
                ticker=ticker,
                training=True
            )

            datasets.append(dataset)
            surviving.append(ticker)

            logger.info("Ticker accepted → %s", ticker)

        except Exception as e:

            logger.warning(
                "Ticker rejected → %s | %s",
                ticker,
                str(e)
            )

    if not datasets:
        raise RuntimeError("All tickers failed — training aborted.")

    survival_ratio = len(surviving) / max(len(universe), 1)

    logger.info("Universe survival ratio → %.2f", survival_ratio)

    df = pd.concat(datasets, ignore_index=True, copy=False)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError("Training aborted — dataset too small.")

    df = cross_sectional_normalize(df)

    validate_feature_schema(df.loc[:, MODEL_FEATURES])

    if df["target"].nunique() < 2:
        raise RuntimeError("Target collapsed.")

    hash_df = df[
        ["ticker", "date", "target", *MODEL_FEATURES]
    ].sort_values(["date", "ticker"]).reset_index(drop=True)

    dataset_hash = MetadataManager.hash_list([
        MetadataManager.fingerprint_dataset(hash_df),
        get_schema_signature(),
        list(MODEL_FEATURES),
        start_date,
        end_date
    ])

    logger.info("Dataset ready | rows=%s", len(df))

    return df, dataset_hash


############################################################
# MAIN
############################################################

def main(start_date=None, end_date=None):

    t0 = time.time()

    init_env()
    enforce_determinism()

    logger.info("===================================")
    logger.info("INSTITUTIONAL XGBOOST TRAINING")
    logger.info("===================================")

    if not start_date or not end_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    df, dataset_hash = load_training_data(start_date, end_date)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ########################################################
    # TRAINER
    ########################################################

    def trainer(d):

        y = d["target"]

        model = build_xgboost_model(y)
        model.fit(d.loc[:, MODEL_FEATURES], y)

        return model

    def signal(model, test):

        probs = model.predict_proba(
            test.loc[:, MODEL_FEATURES]
        )[:, 1]

        return [
            "BUY" if p > BUY_THRESHOLD
            else "SELL" if p < SELL_THRESHOLD
            else "HOLD"
            for p in probs
        ]

    wf = WalkForwardValidator(
        model_trainer=trainer,
        signal_generator=signal
    )

    logger.info("Running walk-forward validation...")

    strategy_metrics = wf.run(df)

    numeric_metrics = sanitize_metrics(strategy_metrics)

    logger.info(
        "Walk-forward complete | Sharpe=%.3f | Return=%.3f | Drawdown=%.3f",
        numeric_metrics.get("avg_sharpe", -1),
        numeric_metrics.get("avg_strategy_return", -1),
        numeric_metrics.get("max_drawdown", -1),
    )

    gc.collect()

    ########################################################
    # FINAL MODEL
    ########################################################

    logger.info("Training final production model...")

    final_model = build_final_xgboost_model(df["target"])

    final_model.fit(
        df.loc[:, MODEL_FEATURES],
        df["target"]
    )

    logger.info("Saving model artifact...")

    size = save_model_atomic(final_model, TEMP_MODEL_PATH)

    logger.info(
        "Model saved successfully | size=%.2f MB",
        size / (1024 * 1024)
    )

    logger.info("Creating metadata...")

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics=numeric_metrics,
        features=list(MODEL_FEATURES),
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        dataset_rows=len(df),
        metadata_type="training_manifest_v1"
    )

    MetadataManager.save_metadata(
        metadata,
        TEMP_METADATA_PATH
    )

    logger.info("Registering model...")

    version = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    logger.info("Model registered | version=%s", version)

    if PROMOTE_FLAG:

        logger.info("Promoting model to production...")

        ModelRegistry.promote_to_production(
            MODEL_DIR,
            version
        )

    else:
        logger.info("Model kept as CANDIDATE.")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    logger.info("===================================")
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("Version → %s", version)
    logger.info("Sharpe → %.3f", numeric_metrics.get("avg_sharpe", -1))
    logger.info("Total time → %.2f minutes", (time.time() - t0) / 60)
    logger.info("===================================")

    return numeric_metrics


if __name__ == "__main__":
    main()
