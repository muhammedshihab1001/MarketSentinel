import os
import time
import joblib
import logging
import random
import hashlib
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
MAX_REASONABLE_SHARPE = 5.0


############################################################
# DETERMINISM (HARDENED)
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# DATASET HASH (STABLE ORDER)
############################################################

def compute_dataset_hash(df: pd.DataFrame) -> str:
    df_sorted = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return MetadataManager.fingerprint_dataset(df_sorted)


############################################################
# METRIC SANITIZATION
############################################################

def sanitize_metrics(metrics: dict) -> dict:

    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError("Walk-forward metrics must be a non-empty dict.")

    sanitized = {}

    for k, v in metrics.items():

        if isinstance(v, (list, tuple, dict, np.ndarray)):
            continue

        val = float(v)

        if not np.isfinite(val):
            raise RuntimeError(f"Non-finite metric detected: {k}")

        sanitized[k] = val

    if not sanitized:
        raise RuntimeError("All metrics were dropped during sanitization.")

    if sanitized.get("avg_sharpe", 0) > MAX_REASONABLE_SHARPE:
        logger.warning("Unusually high Sharpe detected in research phase.")

    return sanitized


############################################################
# LOAD RAW TRAINING DATA
############################################################

def load_training_data(start_date, end_date):

    market_data = MarketDataService()
    store = FeatureStore()
    universe = MarketUniverse.get_universe()

    logger.info("Universe version: %s | size=%s",
                MarketUniverse.get_version(),
                len(universe))

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

    if not df["date"].is_monotonic_increasing:
        raise RuntimeError("Chronological integrity failure.")

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate feature columns detected.")

    return df


############################################################
# FINAL TARGET (LEAKAGE SAFE)
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

    if df["date"].max() <= df["date"].min():
        raise RuntimeError("Date range invalid after target creation.")

    return df


############################################################
# TRAINER (FOLD-ROBUST)
############################################################

def trainer(train_df):

    X = train_df.loc[:, MODEL_FEATURES].copy()

    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=True) <= 1
    ]

    if constant_cols:
        logger.warning(
            "Fold-local constant features neutralized (set to 0): %s",
            constant_cols
        )

        for col in constant_cols:
            X[col] = 0.0

    X = validate_feature_schema(X, mode="training")

    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)

    return pipeline


############################################################
# FINAL TRAINER (STRICT CONTRACT + CONSISTENT WITH CV)
############################################################

def final_trainer(train_df):

    df = train_df.copy()

    # Ensure all required model features exist
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = df.loc[:, MODEL_FEATURES].copy()

    ########################################################
    # NEUTRALIZE CONSTANT FEATURES (CONSISTENT WITH CV)
    ########################################################

    constant_cols = [
        col for col in df.columns
        if df[col].nunique(dropna=True) <= 1
    ]

    if constant_cols:
        logger.warning(
            "Final-train constant features neutralized (set to 0): %s",
            constant_cols
        )

        for col in constant_cols:
            df[col] = 0.0

    ########################################################
    # STRICT SCHEMA VALIDATION
    ########################################################

    X = validate_feature_schema(df, mode="strict_contract")

    if X.isnull().any().any():
        raise RuntimeError("NaN detected before final training.")

    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)

    return pipeline


############################################################
# ARTIFACT EXPORT (ATOMIC SAFE)
############################################################

def export_artifacts(model, metrics, dataset_hash,
                     start_date, end_date, final_df):

    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = int(time.time())

    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
    tmp_path = os.path.join(MODEL_DIR, f"tmp_model_{timestamp}.pkl")

    joblib.dump(model, tmp_path)
    os.replace(tmp_path, model_path)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost",
        metrics=metrics,
        features=tuple(MODEL_FEATURES),
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
    research_metrics = validator.run(raw_df.copy())

    production_metrics = sanitize_metrics(research_metrics)

    final_df = build_final_target(raw_df)

    dataset_hash = compute_dataset_hash(final_df)

    final_model = final_trainer(final_df)

    export_artifacts(
        final_model,
        production_metrics,
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

    return research_metrics


if __name__ == "__main__":
    main()