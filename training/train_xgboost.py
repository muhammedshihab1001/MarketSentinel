# ==========================================================
# TRAIN XGBOOST REGRESSION (Institutional Hardened v3)
# Adaptive Governance | Research + Production Modes
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
import tempfile

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
BASELINE_CONTRACT = os.path.join(MODEL_DIR, "baseline_contract.json")

SEED = 42

STRICT_GOVERNANCE = os.getenv("TRAINING_STRICT_GOVERNANCE", "1") == "1"
TRAINING_MODE = os.getenv("TRAINING_MODE", "research").lower()

MIN_TRAINING_ROWS = 1500
MIN_UNIQUE_DATES = 300
MIN_CS_WIDTH = 10

# Hard leakage protection (never relaxed)
MAX_REASONABLE_SHARPE = 5.0
MAX_PROFIT_FACTOR = 10.0
MIN_WINDOWS = 3

# Production thresholds
MIN_PRODUCTION_SHARPE = 0.10
MAX_DRAWDOWN = -0.55

# Research thresholds (Yahoo-friendly)
MIN_RESEARCH_SHARPE = -0.05
MAX_RESEARCH_DRAWDOWN = -0.75


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
# HASH HELPERS (UNCHANGED)
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
# METRIC VALIDATION (UPDATED + STABLE)
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
    num_windows = int(metrics["num_windows"])

    logger.info(
        "Validation | Mode=%s | Sharpe=%.4f | DD=%.2f%% | PF=%.2f | Windows=%d",
        TRAINING_MODE,
        sharpe,
        drawdown * 100,
        profit_factor,
        num_windows
    )

    # --------------------------
    # Always enforce leakage rules
    # --------------------------

    if sharpe > MAX_REASONABLE_SHARPE:
        raise RuntimeError("Sharpe unrealistically high — leakage suspected.")

    if profit_factor > MAX_PROFIT_FACTOR:
        raise RuntimeError("Profit factor unrealistic — leakage suspected.")

    if num_windows < MIN_WINDOWS:
        raise RuntimeError("Insufficient walk-forward windows.")

    if metrics["final_equity"] <= 0:
        raise RuntimeError("Backtest produced non-positive equity.")

    # --------------------------
    # Mode-dependent validation
    # --------------------------

    if STRICT_GOVERNANCE or TRAINING_MODE == "production":

        if sharpe < MIN_PRODUCTION_SHARPE:
            raise RuntimeError(
                f"Model rejected — Sharpe too low ({sharpe:.4f})."
            )

        if drawdown < MAX_DRAWDOWN:
            raise RuntimeError(
                f"Model rejected — drawdown breach ({drawdown:.2%})."
            )

    else:
        # Research Mode

        if sharpe < MIN_RESEARCH_SHARPE:
            raise RuntimeError(
                f"Research model unstable — Sharpe too low ({sharpe:.4f})."
            )

        if drawdown < MAX_RESEARCH_DRAWDOWN:
            raise RuntimeError(
                f"Research model drawdown extreme ({drawdown:.2%})."
            )


# ==========================================================
# DATA LOADING (UNCHANGED)
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
# TARGET (UNCHANGED)
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
# TRAINERS (UNCHANGED)
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


def final_trainer(train_df):
    return trainer(train_df)


# ==========================================================
# MAIN (UNCHANGED STRUCTURE)
# ==========================================================

def main(create_baseline=False,
         promote_baseline=False,
         allow_soft_fail=False):

    init_env()
    enforce_determinism()

    start_date, end_date = MarketTime.window_for("xgboost")

    try:

        raw_df = load_training_data(start_date, end_date)

        validator = WalkForwardValidator(trainer)
        research_metrics = validator.run(raw_df.copy())

        validate_production_metrics(research_metrics)

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

    except Exception as e:
        if allow_soft_fail:
            logger.warning("Training soft-failed: %s", str(e))
        else:
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
        allow_soft_fail=args.allow_soft_fail
    )