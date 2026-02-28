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
from core.monitoring.drift_detector import DriftDetector
from core.artifacts.metadata_manager import MetadataManager

from training.backtesting.walk_forward import WalkForwardValidator, FORWARD_DAYS
from core.models.xgboost import build_xgboost_pipeline

logger = logging.getLogger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")
PRODUCTION_POINTER = "production_pointer.json"
BASELINE_CONTRACT = os.path.join(MODEL_DIR, "baseline_contract.json")

SEED = 42
MIN_TRAINING_ROWS = 1200
MIN_UNIQUE_DATES = 250

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
        "Production validation | Sharpe=%.4f | DD=%.2f%% | PF=%.2f",
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
# BASELINE GOVERNANCE
# ==========================================================

def save_baseline_contract(metrics, dataset_hash, model_version):
    os.makedirs(MODEL_DIR, exist_ok=True)

    contract = {
        "avg_sharpe": metrics["avg_sharpe"],
        "profit_factor": metrics["profit_factor"],
        "max_drawdown": metrics["max_drawdown"],
        "dataset_hash": dataset_hash,
        "model_version": model_version,
        "feature_checksum": compute_feature_checksum(),
        "created_utc": int(time.time())
    }

    with open(BASELINE_CONTRACT, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    logger.info("Baseline contract saved.")


def load_baseline_contract():
    if not os.path.exists(BASELINE_CONTRACT):
        return None
    with open(BASELINE_CONTRACT, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_against_baseline(metrics):
    baseline = load_baseline_contract()
    if not baseline:
        logger.info("No baseline contract found. Skipping baseline comparison.")
        return

    sharpe_ratio = metrics["avg_sharpe"] / baseline["avg_sharpe"]
    pf_ratio = metrics["profit_factor"] / baseline["profit_factor"]

    if sharpe_ratio < 0.8:
        raise RuntimeError("Sharpe degraded >20% vs baseline.")

    if pf_ratio < 0.8:
        raise RuntimeError("Profit factor degraded >20% vs baseline.")

    if metrics["max_drawdown"] < baseline["max_drawdown"] - 0.10:
        raise RuntimeError("Drawdown materially worse vs baseline.")

    logger.info("Baseline comparison passed.")


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

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate columns detected.")

    validate_feature_schema(df.loc[:, MODEL_FEATURES], mode="training")

    return df


# ==========================================================
# TARGET
# ==========================================================

def build_target(df: pd.DataFrame):

    df = df.sort_values(["ticker", "date"]).copy()

    df["target"] = (
        df.groupby("ticker")["close"]
        .transform(lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x))
    )

    df = df.dropna(subset=["target"])

    MIN_CS_WIDTH = 8
    counts = df.groupby("date")["ticker"].transform("count")
    df = df[counts >= MIN_CS_WIDTH]

    if df.empty:
        raise RuntimeError("All dates dropped after cross-sectional enforcement.")

    logger.info(
        "Regression training dataset | rows=%s | unique_dates=%s",
        len(df),
        df["date"].nunique()
    )

    return df


# ==========================================================
# TRAINERS
# ==========================================================

def trainer(train_df):
    train_df = build_target(train_df)

    X = validate_feature_schema(train_df.loc[:, MODEL_FEATURES], mode="training")
    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)
    return pipeline


def final_trainer(train_df):
    train_df = build_target(train_df)

    X = validate_feature_schema(train_df.loc[:, MODEL_FEATURES], mode="strict_contract")
    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)
    return pipeline


# ==========================================================
# EXPORT
# ==========================================================

def export_artifacts(model, metrics, dataset_hash,
                     start_date, end_date, final_df):

    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = int(time.time())
    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
    tmp_path = os.path.join(MODEL_DIR, f"tmp_model_{timestamp}.pkl")

    joblib.dump(model, tmp_path)
    os.replace(tmp_path, model_path)

    artifact_hash = MetadataManager.hash_file(model_path)

    scalar_metrics = {k: v for k, v in metrics.items() if k != "equity_curve"}

    extra_fields = {
        "artifact_hash": artifact_hash,
        "equity_curve": metrics.get("equity_curve", [])
    }

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
        extra_fields=extra_fields
    )

    metadata_path = os.path.join(MODEL_DIR, f"metadata_{timestamp}.json")
    MetadataManager.save_metadata(metadata, metadata_path)

    pointer_path = os.path.join(MODEL_DIR, PRODUCTION_POINTER)

    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": str(timestamp),
            "updated_at": int(time.time())
        }, f, indent=2)

    logger.info("Artifacts exported safely.")
    return timestamp


# ==========================================================
# MAIN
# ==========================================================

def main(start_date=None, end_date=None,
         create_baseline=False,
         promote_baseline=False,
         allow_soft_fail=False):

    t0 = time.time()

    init_env()
    enforce_determinism()

    if not start_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    raw_df = load_training_data(start_date, end_date)

    validator = WalkForwardValidator(trainer)
    research_metrics = validator.run(raw_df.copy())

    logger.info("Research metrics: %s", research_metrics)

    try:
        validate_production_metrics(research_metrics)
        validate_against_baseline(research_metrics)
    except RuntimeError as e:
        if allow_soft_fail:
            logger.warning("Soft-fail enabled: %s", str(e))
        else:
            raise

    final_df = build_target(raw_df)
    dataset_hash = compute_dataset_hash(final_df)
    final_model = final_trainer(raw_df)

    version = export_artifacts(
        final_model,
        research_metrics,
        dataset_hash,
        start_date,
        end_date,
        final_df
    )

    drift = DriftDetector()

    if create_baseline:
        if os.path.exists(BASELINE_CONTRACT):
            raise RuntimeError("Baseline already exists. Use --promote-baseline.")
        save_baseline_contract(research_metrics, dataset_hash, version)
        drift.create_baseline(
            dataset=final_df.loc[:, MODEL_FEATURES],
            dataset_hash=dataset_hash,
            training_code_hash=MetadataManager.fingerprint_training_code(),
            feature_checksum=compute_feature_checksum(),
            model_version=str(version),
            allow_overwrite=False
        )

    if promote_baseline:
        save_baseline_contract(research_metrics, dataset_hash, version)
        drift.create_baseline(
            dataset=final_df.loc[:, MODEL_FEATURES],
            dataset_hash=dataset_hash,
            training_code_hash=MetadataManager.fingerprint_training_code(),
            feature_checksum=compute_feature_checksum(),
            model_version=str(version),
            allow_overwrite=True
        )
        logger.warning("Baseline PROMOTED (overwrite executed).")

    logger.info("Training completed in %.2f minutes",
                (time.time() - t0) / 60)

    return research_metrics


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