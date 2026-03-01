# ==========================================================
# TRAIN XGBOOST REGRESSION (Clean Quant Research Version)
# CV-Ready | Walk-Forward Validated | Reproducible
# ==========================================================

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
import inspect

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature,
    SCHEMA_VERSION,
    schema_snapshot
)
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse
from core.artifacts.metadata_manager import MetadataManager
from training.backtesting.walk_forward import WalkForwardValidator, FORWARD_DAYS
from core.models.xgboost import build_xgboost_pipeline

logger = logging.getLogger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")
PRODUCTION_POINTER = "production_pointer.json"

SEED = 42
MIN_TRAINING_ROWS = 1200
MIN_UNIQUE_DATES = 250
MIN_CS_WIDTH = 8

MAX_REASONABLE_SHARPE = 5.0
MAX_PROFIT_FACTOR = 10.0
MIN_WINDOWS = 3


# ==========================================================
# DETERMINISM
# ==========================================================

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


# ==========================================================
# HASH HELPERS
# ==========================================================

def compute_dataset_hash(df: pd.DataFrame) -> str:
    df_sorted = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return MetadataManager.fingerprint_dataset(df_sorted)


def compute_feature_checksum():
    return MetadataManager.fingerprint_features(tuple(MODEL_FEATURES))


def compute_training_code_hash():
    source = inspect.getsource(build_xgboost_pipeline)
    return hashlib.sha256(source.encode()).hexdigest()


def compute_reproducibility_hash(dataset_hash):
    payload = {
        "dataset_hash": dataset_hash,
        "schema_signature": get_schema_signature(),
        "schema_version": SCHEMA_VERSION,
        "feature_checksum": compute_feature_checksum(),
        "universe_hash": MarketUniverse.fingerprint(),
        "training_code_hash": compute_training_code_hash()
    }
    canonical = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


# ==========================================================
# ARTIFACT EXPORT
# ==========================================================

def export_artifacts(model,
                     metrics,
                     dataset_hash,
                     start_date,
                     end_date,
                     final_df,
                     create_baseline=False,
                     promote_baseline=False):

    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    version = f"{timestamp}"

    model_path = os.path.join(MODEL_DIR, f"model_{version}.pkl")
    metadata_path = os.path.join(MODEL_DIR, f"metadata_{version}.json")

    joblib.dump(model, model_path)

    artifact_hash = MetadataManager.fingerprint_file(model_path)

    metadata = {
        "model_version": version,
        "created_at": timestamp,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "schema_version": SCHEMA_VERSION,
        "schema_signature": get_schema_signature(),
        "features": list(MODEL_FEATURES),
        "dataset_hash": dataset_hash,
        "artifact_hash": artifact_hash,
        "feature_checksum": compute_feature_checksum(),
        "training_code_hash": compute_training_code_hash(),
        "universe_hash": MarketUniverse.fingerprint(),
        "reproducibility_hash": compute_reproducibility_hash(dataset_hash),
        "metrics": metrics,
        "schema_snapshot": schema_snapshot()
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts saved | version=%s", version)

    # ------------------------------------------------------
    # Promote to production pointer
    # ------------------------------------------------------

    if promote_baseline:
        pointer_path = os.path.join(MODEL_DIR, PRODUCTION_POINTER)

        pointer_data = {
            "model_version": version,
            "promoted_at": timestamp
        }

        with open(pointer_path, "w", encoding="utf-8") as f:
            json.dump(pointer_data, f, indent=2)

        logger.info("Model promoted to production.")


# ==========================================================
# DATA LOADING
# ==========================================================

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
    df = FeatureEngineer.add_cross_sectional_features(df)
    df = FeatureEngineer.finalize(df)

    validate_feature_schema(df.loc[:, MODEL_FEATURES], mode="training")

    return df


# ==========================================================
# TARGET
# ==========================================================

def build_target(df: pd.DataFrame):

    df = df.sort_values(["date", "ticker"]).copy()

    df["raw_forward"] = (
        df.groupby("ticker")["close"]
        .transform(lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x))
    )

    df = df.dropna(subset=["raw_forward"])

    cs_mean = df.groupby("date")["raw_forward"].transform("mean")
    cs_std = df.groupby("date")["raw_forward"].transform("std").replace(0, np.nan)

    df["target"] = (df["raw_forward"] - cs_mean) / cs_std
    df = df[np.isfinite(df["target"])]

    counts = df.groupby("date")["ticker"].transform("count")
    df = df[counts >= MIN_CS_WIDTH]

    df.drop(columns=["raw_forward"], inplace=True)

    if df.empty:
        raise RuntimeError("All dates removed after CS filtering.")

    return df


# ==========================================================
# TRAINER
# ==========================================================

def trainer(train_df):

    train_df = build_target(train_df)
    train_df = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    X = validate_feature_schema(
        train_df.loc[:, MODEL_FEATURES],
        mode="training"
    )

    y = train_df["target"].values

    pipeline = build_xgboost_pipeline()
    pipeline.fit(X, y)

    if list(pipeline.feature_names) != list(MODEL_FEATURES):
        raise RuntimeError("Feature order mismatch after training.")

    return pipeline


# ==========================================================
# MAIN
# ==========================================================

def main(promote_baseline=False, allow_soft_fail=False):

    init_env()
    enforce_determinism()

    start_date, end_date = MarketTime.window_for("xgboost")

    try:

        raw_df = load_training_data(start_date, end_date)

        validator = WalkForwardValidator(trainer)
        metrics = validator.run(raw_df.copy())

        final_df = build_target(raw_df)
        dataset_hash = compute_dataset_hash(final_df)
        final_model = trainer(raw_df)

        export_artifacts(
            final_model,
            metrics,
            dataset_hash,
            start_date,
            end_date,
            final_df,
            promote_baseline=promote_baseline
        )

        logger.info("Training completed successfully.")

    except Exception as e:
        if allow_soft_fail:
            logger.warning("Training soft-failed: %s", str(e))
        else:
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote-baseline", action="store_true")
    parser.add_argument("--allow-soft-fail", action="store_true")
    args = parser.parse_args()

    main(
        promote_baseline=args.promote_baseline,
        allow_soft_fail=args.allow_soft_fail
    )