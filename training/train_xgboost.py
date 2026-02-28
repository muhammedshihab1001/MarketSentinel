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
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
)
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse
from core.artifacts.metadata_manager import MetadataManager

from training.backtesting.walk_forward import WalkForwardValidator, FORWARD_DAYS
from core.models.xgboost import build_xgboost_pipeline

logger = logging.getLogger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")
PRODUCTION_POINTER = "production_pointer.json"
BASELINE_CONTRACT = os.path.join(MODEL_DIR, "baseline_contract.json")

SEED = 42
MIN_TRAINING_ROWS = 1500
MIN_UNIQUE_DATES = 300
MIN_CS_WIDTH = 10

MIN_PRODUCTION_SHARPE = 0.10
MAX_DRAWDOWN = -0.55
MAX_REASONABLE_SHARPE = 5.0
MAX_PROFIT_FACTOR = 10.0


# ==========================================================
# DETERMINISM
# ==========================================================

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)


# ==========================================================
# PRODUCTION METRIC VALIDATION
# ==========================================================

def validate_production_metrics(metrics: dict):

    required = [
        "avg_sharpe",
        "max_drawdown",
        "profit_factor",
        "final_equity",
        "num_windows"
    ]

    for key in required:
        if key not in metrics:
            raise RuntimeError(f"Missing metric: {key}")
        if not np.isfinite(metrics[key]):
            raise RuntimeError(f"Non-finite metric: {key}")

    sharpe = float(metrics["avg_sharpe"])
    drawdown = float(metrics["max_drawdown"])
    profit_factor = float(metrics["profit_factor"])

    logger.info(
        "Validation | Sharpe=%.4f | DD=%.2f%% | PF=%.2f",
        sharpe,
        drawdown * 100,
        profit_factor
    )

    if sharpe > MAX_REASONABLE_SHARPE:
        raise RuntimeError("Sharpe unrealistically high — leakage suspected.")

    if profit_factor > MAX_PROFIT_FACTOR:
        raise RuntimeError("Profit factor unrealistic — leakage suspected.")

    if sharpe < MIN_PRODUCTION_SHARPE:
        raise RuntimeError(f"Model rejected — Sharpe too low ({sharpe:.4f}).")

    if drawdown < MAX_DRAWDOWN:
        raise RuntimeError(f"Model rejected — drawdown breach ({drawdown:.2%}).")

    if metrics["final_equity"] <= 0:
        raise RuntimeError("Backtest produced non-positive equity.")


# ==========================================================
# FEATURE CHECKSUM
# ==========================================================

def compute_feature_checksum():
    canonical = json.dumps(list(MODEL_FEATURES), sort_keys=False).encode()
    return hashlib.sha256(canonical).hexdigest()


# ==========================================================
# DATASET HASH
# ==========================================================

def compute_dataset_hash(df: pd.DataFrame) -> str:
    df_sorted = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return MetadataManager.fingerprint_dataset(df_sorted)


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

    logger.info(
        "Training dataset loaded | rows=%s | dates=%s",
        len(df),
        df["date"].nunique()
    )

    return df


# ==========================================================
# TARGET (PAIRWISE RANKING COMPATIBLE)
# ==========================================================

def build_target(df: pd.DataFrame):

    df = df.sort_values(["date", "ticker"]).copy()

    df["raw_forward"] = (
        df.groupby("ticker")["close"]
        .transform(lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x))
    )

    df = df.dropna(subset=["raw_forward"])

    # Cross-sectional z-score normalization
    cs_mean = df.groupby("date")["raw_forward"].transform("mean")
    cs_std = df.groupby("date")["raw_forward"].transform("std")
    cs_std = cs_std.replace(0, np.nan)

    df["target"] = (df["raw_forward"] - cs_mean) / cs_std
    df = df[np.isfinite(df["target"])]

    counts = df.groupby("date")["ticker"].transform("count")
    df = df[counts >= MIN_CS_WIDTH]

    df.drop(columns=["raw_forward"], inplace=True)

    if df.empty:
        raise RuntimeError("All dates removed after CS filtering.")

    return df


def build_groups(df: pd.DataFrame):

    groups = df.groupby("date").size().values.astype(int)

    if min(groups) < MIN_CS_WIDTH:
        raise RuntimeError("Group size violation.")

    return groups


# ==========================================================
# TRAINERS
# ==========================================================

def trainer(train_df):

    if "target" not in train_df.columns:
        train_df = build_target(train_df)

    train_df = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    X = validate_feature_schema(
        train_df.loc[:, MODEL_FEATURES],
        mode="training"
    )

    y = train_df["target"].values
    groups = build_groups(train_df)

    pipeline = build_xgboost_pipeline()
    pipeline.fit(X, y, groups)

    return pipeline


def final_trainer(train_df):

    train_df = build_target(train_df)
    train_df = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    X = validate_feature_schema(
        train_df.loc[:, MODEL_FEATURES],
        mode="strict_contract"
    )

    y = train_df["target"].values
    groups = build_groups(train_df)

    pipeline = build_xgboost_pipeline()
    pipeline.fit(X, y, groups)

    return pipeline


# ==========================================================
# EXPORT
# ==========================================================

def export_artifacts(model, metrics, dataset_hash,
                     start_date, end_date, final_df,
                     create_baseline=False,
                     promote_baseline=False):

    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = int(time.time())
    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")

    joblib.dump(model, model_path)

    artifact_hash = MetadataManager.hash_file(model_path)

    scalar_metrics = {k: v for k, v in metrics.items() if k != "equity_curve"}

    metadata = MetadataManager.create_metadata(
        model_name="xgboost",
        metrics=scalar_metrics,
        features=tuple(MODEL_FEATURES),
        training_start=str(start_date),
        training_end=str(end_date),
        dataset_hash=dataset_hash,
        dataset_rows=len(final_df),
        metadata_type="training_manifest_v1",
        feature_checksum=compute_feature_checksum(),
        extra_fields={
            "artifact_hash": artifact_hash
        }
    )

    metadata_path = os.path.join(MODEL_DIR, f"metadata_{timestamp}.json")
    MetadataManager.save_metadata(metadata, metadata_path)

    if promote_baseline:
        pointer_path = os.path.join(MODEL_DIR, PRODUCTION_POINTER)
        with open(pointer_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_version": str(timestamp),
                "updated_at": int(time.time())
            }, f, indent=2)
        logger.info("Model promoted to production.")

    return timestamp


# ==========================================================
# MAIN
# ==========================================================

def main(create_baseline=False,
         promote_baseline=False,
         allow_soft_fail=False):

    init_env()
    enforce_determinism()

    start_date, end_date = MarketTime.window_for("xgboost")

    raw_df = load_training_data(start_date, end_date)

    validator = WalkForwardValidator(trainer)
    research_metrics = validator.run(raw_df.copy())

    logger.info("Research metrics: %s", research_metrics)

    try:
        validate_production_metrics(research_metrics)
    except RuntimeError as e:
        if allow_soft_fail:
            logger.warning("Soft fail: %s", str(e))
        else:
            raise

    final_df = build_target(raw_df)
    dataset_hash = compute_dataset_hash(final_df)
    final_model = final_trainer(raw_df)

    export_artifacts(
        final_model,
        research_metrics,
        dataset_hash,
        start_date,
        end_date,
        final_df,
        create_baseline=create_baseline,
        promote_baseline=promote_baseline
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-baseline", action="store_true")
    parser.add_argument("--promote-baseline", action="store_true")
    parser.add_argument("--allow-soft-fail", action="store_true")

    args = parser.parse_args()

    main(
        create_baseline=args.create_baseline,
        promote_baseline=args.promote_baseline,
        allow_soft_fail=args.allow_soft_fail
    )