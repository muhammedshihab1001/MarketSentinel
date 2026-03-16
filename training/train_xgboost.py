# ==========================================================
# TRAIN XGBOOST REGRESSION (Hybrid + Baseline Enabled v2.4)
# CV-Ready | Walk-Forward Validated | Drift Governance
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

logger = logging.getLogger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")
DRIFT_DIR = os.path.abspath("artifacts/drift")
BASELINE_PATH = os.path.join(DRIFT_DIR, "baseline.json")
PRODUCTION_POINTER = "production_pointer.json"

SEED = 42

MIN_TRAINING_ROWS = 1200
MIN_UNIQUE_DATES = 250
MIN_CS_WIDTH = 8
TARGET_CLIP = 5.0

LOW_VARIANCE_THRESHOLD = 1e-6
MAX_DATASET_ROWS = 1_000_000

# NEW: allow partial universe failure
MIN_SUCCESSFUL_TICKERS = 8


# ==========================================================
# DETERMINISM
# ==========================================================

def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)


# ==========================================================
# HASH UTILITIES
# ==========================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def compute_dataset_hash(df: pd.DataFrame):

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
# BASELINE BUILDER
# ==========================================================

def build_baseline_from_dataframe(df, model_version, dataset_hash):

    os.makedirs(DRIFT_DIR, exist_ok=True)

    feature_block = df.loc[:, MODEL_FEATURES].copy()

    features_payload = {}

    for col in MODEL_FEATURES:

        series = pd.to_numeric(feature_block[col], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()

        if len(series) < 50:
            continue

        mean = float(series.mean())
        std = float(series.std())

        try:
            counts, bins = np.histogram(series, bins=10)
        except Exception:
            continue

        features_payload[col] = {
            "mean": mean,
            "std": std,
            "variance": float(series.var()),
            "bin_edges": bins.tolist(),
            "expected_counts": counts.tolist(),
        }

    baseline = {
        "meta": {
            "baseline_version": DriftDetector.BASELINE_VERSION,
            "schema_signature": get_schema_signature(),
            "schema_version": SCHEMA_VERSION,
            "feature_checksum": compute_feature_checksum(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_hash": dataset_hash,
            "model_version": model_version,
            "universe_hash": MarketUniverse.fingerprint(),
        },
        "features": features_payload,
    }

    clone = dict(baseline)

    canonical = json.dumps(clone, sort_keys=True).encode()

    baseline["integrity_hash"] = hashlib.sha256(canonical).hexdigest()

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    logger.info("Baseline created/updated.")


# ==========================================================
# ARTIFACT EXPORT
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

    version = time.strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(MODEL_DIR, f"model_{version}.pkl")

    metadata_path = os.path.join(MODEL_DIR, f"metadata_{version}.json")

    joblib.dump(model, model_path)

    artifact_hash = sha256_file(model_path)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_regressor",
        metrics=metrics,
        features=MODEL_FEATURES,
        training_start=str(start_date),
        training_end=str(end_date),
        dataset_hash=dataset_hash,
        dataset_rows=dataset_rows,
        metadata_type="xgboost_model",
        artifact_hash=artifact_hash,
        feature_checksum=compute_feature_checksum(),
        extra_fields={
            "model_version": version,
            "reproducibility_hash": compute_reproducibility_hash(dataset_hash),
        },
    )

    MetadataManager.save_metadata(metadata, metadata_path)

    logger.info("Artifacts saved | version=%s", version)

    if create_baseline or promote_baseline:

        build_baseline_from_dataframe(
            training_df,
            version,
            dataset_hash,
        )

    if promote_baseline:

        pointer_path = os.path.join(MODEL_DIR, PRODUCTION_POINTER)

        with open(pointer_path, "w", encoding="utf-8") as f:

            json.dump(
                {
                    "model_version": version,
                    "promoted_at": version,
                },
                f,
                indent=2,
            )

        logger.info("Model promoted to production.")

    return version


# ==========================================================
# DATA LOADING
# ==========================================================

def load_training_data(start_date, end_date):

    market_data = MarketDataService()

    universe = MarketUniverse.get_universe()

    price_frames = []

    failures = []

    for ticker in universe:

        try:

            df = market_data.get_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if df is not None and not df.empty:

                price_frames.append(df)

        except Exception as exc:

            failures.append(ticker)

            logger.warning("Price fetch failed for %s | %s", ticker, exc)

    if len(price_frames) < MIN_SUCCESSFUL_TICKERS:

        raise RuntimeError(
            f"Too many tickers failed ({len(price_frames)} usable)"
        )

    combined_prices = pd.concat(price_frames, ignore_index=True)

    if len(combined_prices) > MAX_DATASET_ROWS:

        combined_prices = (
            combined_prices
            .sort_values(["date", "ticker"])
            .tail(MAX_DATASET_ROWS)
        )

    df = FeatureEngineer.build_feature_pipeline(
        combined_prices,
        training=True,
    )

    if len(df) < MIN_TRAINING_ROWS:

        raise RuntimeError("Dataset too small after feature engineering.")

    if df["date"].nunique() < MIN_UNIQUE_DATES:

        raise RuntimeError("Not enough unique trading dates.")

    validate_feature_schema(df.loc[:, MODEL_FEATURES], mode="training")

    logger.info(
        "Training dataset ready | rows=%d tickers=%d dates=%d",
        len(df),
        df["ticker"].nunique(),
        df["date"].nunique(),
    )

    return df


# ==========================================================
# TARGET
# ==========================================================

def build_target(df):

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

    counts = df.groupby("date")["ticker"].transform("count")

    df = df[counts >= MIN_CS_WIDTH]

    df.drop(columns=["raw_forward"], inplace=True)

    return df


# ==========================================================
# TRAINER
# ==========================================================

def trainer(train_df):

    train_df = build_target(train_df)

    train_df = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    X = validate_feature_schema(
        train_df.loc[:, MODEL_FEATURES],
        mode="training",
    )

    y = train_df["target"].values

    if np.std(y) < LOW_VARIANCE_THRESHOLD:

        logger.warning("Low target variance — injecting small noise.")

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
    allow_soft_fail=False,
):

    init_env()

    enforce_determinism()

    if start_date is None or end_date is None:

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
            len(final_df),
            start_date,
            end_date,
            final_df,
            promote_baseline,
            create_baseline,
        )

        logger.info("Training completed successfully.")

        return metrics

    except Exception as exc:

        if allow_soft_fail:

            logger.warning("Training soft-failed: %s", exc)

            return {}

        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--create-baseline", action="store_true")

    parser.add_argument("--promote-baseline", action="store_true")

    parser.add_argument("--allow-soft-fail", action="store_true")

    args = parser.parse_args()

    main(
        create_baseline=args.create_baseline,
        promote_baseline=args.promote_baseline,
        allow_soft_fail=args.allow_soft_fail,
    )