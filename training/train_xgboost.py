# ==========================================================
# TRAIN XGBOOST REGRESSION (Hybrid + Baseline Enabled v2.10)
# ==========================================================

import os
import time
import joblib
import random
import numpy as np
import pandas as pd
import argparse
import hashlib
import json
import inspect

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature,
    SCHEMA_VERSION,
)
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse
from core.artifacts.metadata_manager import MetadataManager
from training.backtesting.walk_forward import WalkForwardValidator, FORWARD_DAYS
from core.models.xgboost import build_xgboost_pipeline
from core.monitoring.drift_detector import DriftDetector
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")
DRIFT_DIR = os.path.abspath("artifacts/drift")

BASELINE_PATH = os.path.join(DRIFT_DIR, "baseline.json")
PRODUCTION_POINTER = os.path.join(MODEL_DIR, "production_pointer.json")

SEED = 42

MIN_TRAINING_ROWS = 1200
TARGET_CLIP = 5.0
LOW_VARIANCE_THRESHOLD = 1e-6
MAX_DATASET_ROWS = 1_000_000

MIN_SUCCESSFUL_TICKERS = 8


def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)


def compute_dataset_hash(df: pd.DataFrame):

    if {"date", "ticker"}.issubset(df.columns):

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return MetadataManager.fingerprint_dataset(df)


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
        "training_code_hash": compute_training_code_hash(),
    }

    canonical = json.dumps(payload, sort_keys=True).encode()

    return hashlib.sha256(canonical).hexdigest()


# ==========================================================
# EXPORT ARTIFACTS
# ==========================================================

def export_artifacts(
    model,
    metrics,
    dataset_hash,
    dataset_rows,
    start_date,
    end_date,
    training_df,
    promote_baseline=False,
    create_baseline=False,
):

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DRIFT_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_version = f"xgb_{timestamp}"

    model_path = os.path.join(MODEL_DIR, f"model_{model_version}.pkl")
    metadata_path = os.path.join(MODEL_DIR, f"metadata_{model_version}.json")

    logger.info("Saving model → %s", model_path)

    joblib.dump(model, model_path)

    artifact_hash = MetadataManager.hash_file(model_path)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost",
        metrics=metrics,
        features=MODEL_FEATURES,
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        dataset_rows=dataset_rows,
        metadata_type="model_training",
        artifact_hash=artifact_hash,
        feature_checksum=compute_feature_checksum(),
    )

    MetadataManager.save_metadata(metadata, metadata_path)

    logger.info("Metadata saved.")

    pointer = {
        "model_version": model_version,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with open(PRODUCTION_POINTER, "w") as f:
        json.dump(pointer, f, indent=2)

    logger.info("Production pointer updated.")

    if create_baseline or promote_baseline:

        logger.info("Creating drift baseline.")

        detector = DriftDetector()

        detector.create_baseline(
            dataset=training_df,
            dataset_hash=dataset_hash,
            training_code_hash=compute_training_code_hash(),
            feature_checksum=compute_feature_checksum(),
            model_version=model_version,
            allow_overwrite=True,
        )

    logger.info("Artifact export completed.")

    return model_version


# ==========================================================
# DATA LOADING
# ==========================================================

def load_training_data(start_date, end_date):

    market_data = MarketDataService()

    universe = MarketUniverse.get_universe()

    price_frames = []

    for ticker in universe:

        try:

            df = market_data.get_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                logger.warning(
                    "Empty price data | ticker=%s | function=load_training_data",
                    ticker,
                )
                continue

            if "date" in df.columns:

                df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
                df = df.dropna(subset=["date"])

            price_frames.append(df)

        except Exception as exc:

            logger.warning(
                "Price fetch failed | ticker=%s | error=%s | function=load_training_data",
                ticker,
                exc,
            )

    if len(price_frames) < MIN_SUCCESSFUL_TICKERS:

        raise RuntimeError(
            f"Too many tickers failed ({len(price_frames)} usable)"
        )

    combined_prices = pd.concat(price_frames, ignore_index=True)

    df = FeatureEngineer.build_feature_pipeline(
        combined_prices,
        training=True,
    )

    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    validate_feature_schema(df.loc[:, MODEL_FEATURES], mode="training")

    if len(df) > MAX_DATASET_ROWS:

        logger.warning(
            "Dataset too large — sampling | rows=%d | function=load_training_data",
            len(df),
        )

        df = df.sample(MAX_DATASET_ROWS, random_state=SEED)

    return df


# ==========================================================
# TARGET
# ==========================================================

def build_target(df):

    required = {"date", "ticker", "close"}

    if not required.issubset(df.columns):

        raise ValueError("Target construction requires date/ticker/close")

    df = df.sort_values(["date", "ticker"]).copy()

    df["raw_forward"] = (
        df.groupby("ticker")["close"]
        .transform(lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x))
    )

    df = df.dropna(subset=["raw_forward"])

    cs_mean = df.groupby("date")["raw_forward"].transform("mean")

    cs_std = (
        df.groupby("date")["raw_forward"]
        .transform("std")
        .replace(0, np.nan)
    )

    df["target"] = (df["raw_forward"] - cs_mean) / cs_std

    df["target"] = np.clip(df["target"], -TARGET_CLIP, TARGET_CLIP)

    df = df[np.isfinite(df["target"])]

    df.drop(columns=["raw_forward"], inplace=True)

    return df


# ==========================================================
# TRAINER
# ==========================================================

def trainer(train_df):

    train_df = train_df.copy()

    train_df = train_df.loc[:, ~train_df.columns.duplicated(keep="first")]

    if {"date", "ticker", "close"}.issubset(train_df.columns):

        logger.info("Market dataset detected — building target.")

        train_df = build_target(train_df)

        X = validate_feature_schema(
            train_df.loc[:, MODEL_FEATURES],
            mode="training",
        )

        y = train_df["target"].values

    else:

        logger.info("Synthetic dataset detected — generating dummy target.")

        X = validate_feature_schema(
            train_df.loc[:, MODEL_FEATURES],
            mode="training",
        )

        y = np.random.normal(0, 1, size=len(X))

    if len(X) < MIN_TRAINING_ROWS:

        logger.warning(
            "Small training dataset | rows=%d | function=trainer",
            len(X),
        )

    if np.std(y) < LOW_VARIANCE_THRESHOLD:

        logger.warning("Low target variance — injecting noise.")

        y = y + np.random.normal(0, 1e-4, size=len(y))

    pipeline = build_xgboost_pipeline()

    pipeline.fit(X, y)

    dataset_hash = compute_dataset_hash(train_df)

    pipeline.training_fingerprint = compute_reproducibility_hash(dataset_hash)

    return pipeline


# ==========================================================
# MAIN
# ==========================================================

def main(
    start_date=None,
    end_date=None,
    create_baseline=False,
    promote_baseline=False,
):

    init_env()

    enforce_determinism()

    if start_date is None or end_date is None:
        start_date, end_date = MarketTime.window_for("xgboost")

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
        len(final_df),
        start_date,
        end_date,
        final_df,
        promote_baseline=promote_baseline,
        create_baseline=create_baseline,
    )

    logger.info("Training completed successfully.")

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--create-baseline", action="store_true")
    parser.add_argument("--promote-baseline", action="store_true")

    args = parser.parse_args()

    main(
        create_baseline=args.create_baseline,
        promote_baseline=args.promote_baseline,
    )