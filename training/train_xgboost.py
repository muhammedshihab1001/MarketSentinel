import os
import time
import joblib
import logging
import random
import numpy as np
import pandas as pd
import argparse
import hashlib
import json

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
PRODUCTION_POINTER = "production_pointer.json"

SEED = 42
MIN_TRAINING_ROWS = 1200
MIN_UNIQUE_DATES = 250
MAX_REASONABLE_SHARPE = 5.0


############################################################
# DETERMINISM
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# FEATURE CHECKSUM
############################################################

def compute_feature_checksum():
    canonical = json.dumps(
        list(MODEL_FEATURES),
        sort_keys=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


############################################################
# DATASET HASH
############################################################

def compute_dataset_hash(df: pd.DataFrame) -> str:
    df_sorted = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return MetadataManager.fingerprint_dataset(df_sorted)


############################################################
# METRIC SANITIZATION
############################################################

def sanitize_metrics(metrics: dict) -> dict:

    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError("Walk-forward metrics must be non-empty dict.")

    sanitized = {}

    for k, v in metrics.items():

        if isinstance(v, (list, tuple, dict, np.ndarray)):
            continue

        val = float(v)

        if not np.isfinite(val):
            raise RuntimeError(f"Non-finite metric detected: {k}")

        sanitized[k] = val

    if sanitized.get("avg_sharpe", 0) > MAX_REASONABLE_SHARPE:
        logger.warning("Unusually high Sharpe detected.")

    return sanitized


############################################################
# LOAD DATA
############################################################

def load_training_data(start_date, end_date):

    market_data = MarketDataService()
    store = FeatureStore()
    universe = MarketUniverse.get_universe()

    datasets = []

    for ticker in universe:

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

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate columns detected.")

    _ = validate_feature_schema(
        df.loc[:, MODEL_FEATURES],
        mode="training"
    )

    return df


############################################################
# TARGET
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

    return df


############################################################
# TRAINER
############################################################

def trainer(train_df):

    X = validate_feature_schema(
        train_df.loc[:, MODEL_FEATURES],
        mode="training"
    )

    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)

    return pipeline


############################################################
# FINAL TRAINER
############################################################

def final_trainer(train_df):

    X = validate_feature_schema(
        train_df.loc[:, MODEL_FEATURES],
        mode="strict_contract"
    )

    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)

    return pipeline


############################################################
# UPDATE PRODUCTION POINTER
############################################################

def update_production_pointer(version: str):

    pointer_path = os.path.join(MODEL_DIR, PRODUCTION_POINTER)

    payload = {
        "model_version": str(version),
        "updated_at": int(time.time())
    }

    tmp_path = pointer_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, pointer_path)

    logger.info("Production pointer updated to version=%s", version)


############################################################
# EXPORT
############################################################

def export_artifacts(model, metrics, dataset_hash,
                     start_date, end_date, final_df):

    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = int(time.time())

    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
    tmp_path = os.path.join(MODEL_DIR, f"tmp_model_{timestamp}.pkl")

    joblib.dump(model, tmp_path)
    os.replace(tmp_path, model_path)

    feature_checksum = compute_feature_checksum()

    metadata = MetadataManager.create_metadata(
        model_name="xgboost",
        metrics=metrics,
        features=tuple(MODEL_FEATURES),
        feature_checksum=feature_checksum,
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

    update_production_pointer(timestamp)

    logger.info("Artifacts exported.")

    return timestamp


############################################################
# MAIN
############################################################

def main(start_date=None, end_date=None, create_baseline=False):

    t0 = time.time()

    init_env()
    enforce_determinism()

    if not start_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    raw_df = load_training_data(start_date, end_date)

    validator = WalkForwardValidator(trainer)
    research_metrics = validator.run(raw_df.copy())

    production_metrics = sanitize_metrics(research_metrics)

    final_df = build_final_target(raw_df)

    dataset_hash = compute_dataset_hash(final_df)

    final_model = final_trainer(final_df)

    version = export_artifacts(
        final_model,
        production_metrics,
        dataset_hash,
        start_date,
        end_date,
        final_df
    )

    if create_baseline:

        drift = DriftDetector()

        drift.create_baseline(
            dataset=final_df.loc[:, MODEL_FEATURES],
            dataset_hash=dataset_hash,
            training_code_hash=MetadataManager.fingerprint_training_code(),
            feature_checksum=compute_feature_checksum(),
            model_version=str(version),
            allow_overwrite=False
        )

        logger.info(
            "Drift baseline created for model_version=%s",
            version
        )

    logger.info(
        "Training completed in %.2f minutes",
        (time.time() - t0) / 60
    )

    return research_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-baseline", action="store_true")
    args = parser.parse_args()

    main(create_baseline=args.create_baseline)