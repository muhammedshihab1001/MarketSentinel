import os
import time
import joblib
import logging
import random
import numpy as np
import pandas as pd

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
)
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse
from core.monitoring.drift_detector import DriftDetector
from core.artifacts.metadata_manager import MetadataManager

from training.backtesting.walk_forward import WalkForwardValidator, FORWARD_DAYS
from models.xgboost_model import build_xgboost_pipeline

logger = logging.getLogger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")

SEED = 42
MIN_TRAINING_ROWS = 1200
MIN_UNIQUE_DATES = 250


############################################################
# DETERMINISM
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# DATASET HASH
############################################################

def compute_dataset_hash(df: pd.DataFrame) -> str:
    return MetadataManager.fingerprint_dataset(df)


############################################################
# LOAD RAW TRAINING DATA
############################################################

def load_training_data(start_date, end_date):

    market_data = MarketDataService()
    store = FeatureStore()
    universe = MarketUniverse.get_universe()

    datasets = []

    for ticker in universe:

        logger.info("Building features for %s", ticker)

        price_df = market_data.get_price_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )

        dataset = store.get_features(
            price_df,
            sentiment_df=None,
            ticker=ticker,
            training=False
        )

        if dataset is None or dataset.empty:
            continue

        datasets.append(dataset)

    if not datasets:
        raise RuntimeError("All tickers failed.")

    df = pd.concat(datasets, ignore_index=True)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError("Dataset too small.")

    if df["date"].nunique() < MIN_UNIQUE_DATES:
        raise RuntimeError("Insufficient unique dates.")

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df


############################################################
# FINAL TARGET
############################################################

def build_final_target(df: pd.DataFrame):

    df = df.sort_values(["ticker", "date"]).copy()

    df["forward_log_return"] = (
        df.groupby("ticker")["close"]
        .transform(lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x))
    )

    df = df.dropna(subset=["forward_log_return"])

    df["alpha_rank_pct"] = (
        df.groupby("date")["forward_log_return"]
        .rank(pct=True)
    )

    df["target"] = np.nan
    df.loc[df["alpha_rank_pct"] >= 0.7, "target"] = 1
    df.loc[df["alpha_rank_pct"] <= 0.3, "target"] = 0

    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    if df["target"].nunique() < 2:
        raise RuntimeError("Training labels collapsed.")

    return df


############################################################
# TRAINER
############################################################

def trainer(train_df):

    X = validate_feature_schema(
        train_df.loc[:, MODEL_FEATURES],
        strict=False
    )

    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)

    return pipeline


def final_trainer(train_df):

    df = train_df.copy()

    # Ensure all MODEL_FEATURES exist
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0  # neutral default

    # Ensure column order
    df = df.loc[:, MODEL_FEATURES]

    X = validate_feature_schema(
        df,
        strict=True
    )

    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)

    return pipeline


############################################################
# ARTIFACT EXPORT (INSTITUTIONAL SAFE)
############################################################

def export_artifacts(model, metrics, dataset_hash,
                     start_date, end_date, final_df):

    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = int(time.time())

    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
    tmp_path = os.path.join(MODEL_DIR, f"tmp_model_{timestamp}.pkl")

    joblib.dump(model, tmp_path)
    os.replace(tmp_path, model_path)

    ########################################################
    # 🔒 OFFICIAL METADATA CREATION
    ########################################################

    metadata = MetadataManager.create_metadata(
        model_name="xgboost",
        metrics=metrics,
        features=MODEL_FEATURES,
        training_start=str(start_date),
        training_end=str(end_date),
        dataset_hash=dataset_hash,
        dataset_rows=len(final_df),
        metadata_type="training_manifest_v1"
    )

    metadata_path = os.path.join(
        MODEL_DIR,
        f"metadata_{timestamp}.json"
    )

    MetadataManager.save_metadata(metadata, metadata_path)

    logger.info("Artifacts exported with institutional metadata.")


############################################################
# MAIN
############################################################

def main(start_date=None, end_date=None):

    t0 = time.time()

    init_env()
    enforce_determinism()

    if not start_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    logger.info("Training window | %s -> %s", start_date, end_date)

    raw_df = load_training_data(start_date, end_date)

    validator = WalkForwardValidator(trainer)
    metrics = validator.run(raw_df)

    final_df = build_final_target(raw_df)

    dataset_hash = compute_dataset_hash(final_df)

    final_model = final_trainer(final_df)

    export_artifacts(
        final_model,
        metrics,
        dataset_hash,
        start_date,
        end_date,
        final_df
    )

    drift = DriftDetector()
    drift.create_baseline(
        dataset=final_df,
        dataset_hash=dataset_hash,
        training_code_hash=MetadataManager.fingerprint_training_code(),
        allow_overwrite=True
    )

    logger.info(
        "Training completed in %.2f minutes",
        (time.time() - t0) / 60
    )

    return metrics


if __name__ == "__main__":
    main()