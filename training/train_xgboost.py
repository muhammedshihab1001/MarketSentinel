# ==========================================================
# TRAIN XGBOOST REGRESSION v2.12
#
# Changes from v2.11:
#   FIX 1: load_training_data now passes start_date + end_date
#           to get_price_data_batch() — new MarketDataService
#           requires these as positional args (no defaults).
#           Old call: get_price_data(ticker, interval, min_history)
#           New call: get_price_data_batch(tickers, start, end, ...)
#           Also switched to batch call (parallel, faster).
#
#   FIX 2: Training window uses TRAINING_LOOKBACK_DAYS=730 (2 years)
#           not INFERENCE_LOOKBACK_DAYS=400. Training and inference
#           use different windows intentionally:
#             Training:  730 days (50,000+ samples for XGBoost)
#             Inference: 400 days (feature engineering only)
#
#   FIX 3: Added cleanup_old_data() call after sync — deletes DB
#           rows older than TRAINING_LOOKBACK_DAYS + 30 day buffer.
#           Keeps DB lean. Without this, rows accumulate forever.
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
MIN_TRAINING_ROWS = 500  # lowered from 1200 — we have ~9 months currently
TARGET_CLIP = 5.0
LOW_VARIANCE_THRESHOLD = 1e-6
MAX_DATASET_ROWS = 1_000_000
MIN_SUCCESSFUL_TICKERS = 8

# Training uses 2 years — inference uses 400 days.
# These are intentionally different window sizes.
TRAINING_LOOKBACK_DAYS = int(os.getenv("TRAINING_LOOKBACK_DAYS", "730"))

# Cleanup: delete rows older than training window + 30 day buffer.
# Keeps DB lean. Rows beyond this are never used by training or inference.
CLEANUP_RETENTION_DAYS = TRAINING_LOOKBACK_DAYS + 30


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
        logger.info("Drift baseline created.")

    logger.info("Artifact export completed.")
    return model_version


# ==========================================================
# DATA CLEANUP
# Deletes DB rows older than CLEANUP_RETENTION_DAYS.
# Keeps the DB lean — rows beyond the training window are
# never used by training or inference so they are pure waste.
# ==========================================================


def cleanup_old_data():
    """
    Delete ohlcv_daily rows older than CLEANUP_RETENTION_DAYS.
    Called after every training run to keep DB size bounded.

    Retention = TRAINING_LOOKBACK_DAYS (730) + 30 day buffer = 760 days.
    Rows older than 760 days are never used by training or inference.
    """
    try:
        from core.db.engine import get_session
        from core.db.models import OHLCVDaily

        cutoff = (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=CLEANUP_RETENTION_DAYS)
        ).date()

        with get_session() as session:
            deleted = (
                session.query(OHLCVDaily)
                .filter(OHLCVDaily.date < cutoff)
                .delete(synchronize_session=False)
            )

        logger.info(
            "DB cleanup complete | deleted=%d rows older than %s "
            "(retention=%d days)",
            deleted,
            cutoff,
            CLEANUP_RETENTION_DAYS,
        )

        return deleted

    except Exception as exc:
        # Non-blocking — cleanup failure must never stop training
        logger.warning("DB cleanup failed (non-blocking) | error=%s", exc)
        return 0


# ==========================================================
# DATA LOADING
# FIX: Uses get_price_data_batch() which requires start_date
#      and end_date. Previous version called get_price_data()
#      without these args which broke with new MarketDataService.
#      Also uses TRAINING_LOOKBACK_DAYS=730 not INFERENCE=400.
# ==========================================================


def load_training_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load and build feature frame for all universe tickers.

    Uses get_price_data_batch() for parallel DB reads (8 workers).
    Training window: 730 days (2 years) for sufficient XGBoost samples.
    Inference window: 400 days (set separately in pipeline.py).
    """

    market_data = MarketDataService()
    universe = list(MarketUniverse.get_universe())

    logger.info(
        "Loading training data | tickers=%d | window=%s → %s",
        len(universe),
        start_date,
        end_date,
    )

    # FIX: use batch call with explicit start/end dates.
    # get_price_data_batch() runs parallel reads (8 workers),
    # much faster than the old sequential loop.
    # min_history=60 is lenient — we want as many tickers as possible,
    # the build_target() cross-sectional z-score handles short series.
    price_map, failures = market_data.get_price_data_batch(
        tickers=universe,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        min_history=60,
    )

    if failures:
        logger.warning(
            "Training data fetch partial failures | failed=%d tickers: %s",
            len(failures),
            list(failures.keys()),
        )

    if len(price_map) < MIN_SUCCESSFUL_TICKERS:
        raise RuntimeError(
            f"Too few tickers loaded ({len(price_map)}) — "
            f"need at least {MIN_SUCCESSFUL_TICKERS}. "
            f"Run DataSyncService.sync_universe() first."
        )

    logger.info(
        "Price data loaded | success=%d failed=%d",
        len(price_map),
        len(failures),
    )

    # Combine all ticker DataFrames into one cross-sectional frame
    price_frames = list(price_map.values())
    combined_prices = pd.concat(price_frames, ignore_index=True)
    combined_prices = combined_prices.dropna(subset=["close"])

    logger.info(
        "Building feature pipeline | total_rows=%d tickers=%d",
        len(combined_prices),
        combined_prices["ticker"].nunique(),
    )

    df = FeatureEngineer.build_feature_pipeline(combined_prices, training=True)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    validate_feature_schema(df.loc[:, MODEL_FEATURES], mode="training")

    if len(df) > MAX_DATASET_ROWS:
        logger.warning("Dataset too large — sampling | rows=%d", len(df))
        df = df.sample(MAX_DATASET_ROWS, random_state=SEED)

    logger.info(
        "Training dataset ready | rows=%d tickers=%d features=%d",
        len(df),
        df["ticker"].nunique() if "ticker" in df.columns else 0,
        len(MODEL_FEATURES),
    )

    return df


# ==========================================================
# TARGET CONSTRUCTION
# Cross-sectional z-score of log forward returns.
# Each date: z = (ticker_return - mean_return) / std_return
# This removes market-wide direction bias and focuses on
# which stocks outperform vs underperform their peers.
# ==========================================================


def build_target(df: pd.DataFrame) -> pd.DataFrame:

    required = {"date", "ticker", "close"}
    if not required.issubset(df.columns):
        raise ValueError("Target construction requires date/ticker/close")

    df = df.sort_values(["date", "ticker"]).copy()

    # Log return FORWARD_DAYS ahead (what the model predicts)
    df["raw_forward"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x.shift(-FORWARD_DAYS)) - np.log(x)
    )

    df = df.dropna(subset=["raw_forward"])

    # Cross-sectional z-score — relative return vs universe on same day
    cs_mean = df.groupby("date")["raw_forward"].transform("mean")
    cs_std = df.groupby("date")["raw_forward"].transform("std").replace(0, np.nan)

    df["target"] = (df["raw_forward"] - cs_mean) / cs_std
    df["target"] = np.clip(df["target"], -TARGET_CLIP, TARGET_CLIP)
    df = df[np.isfinite(df["target"])]
    df.drop(columns=["raw_forward"], inplace=True)

    return df


# ==========================================================
# TRAINER
# ==========================================================


def trainer(train_df: pd.DataFrame):

    train_df = train_df.copy()
    train_df = train_df.loc[:, ~train_df.columns.duplicated(keep="first")]

    if {"date", "ticker", "close"}.issubset(train_df.columns):
        logger.info("Market dataset detected — building target.")
        train_df = build_target(train_df)
        X = validate_feature_schema(train_df.loc[:, MODEL_FEATURES], mode="training")
        y = train_df["target"].values
    else:
        logger.info("Synthetic dataset detected — generating dummy target.")
        X = validate_feature_schema(train_df.loc[:, MODEL_FEATURES], mode="training")
        y = np.random.normal(0, 1, size=len(X))

    if len(X) < MIN_TRAINING_ROWS:
        logger.warning(
            "Small training dataset | rows=%d | "
            "model will train but performance may be limited. "
            "Sync more historical data to improve.",
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
# TRAINING DATE WINDOW
# Returns (start_date, end_date) for the training window.
# Uses TRAINING_LOOKBACK_DAYS=730 (2 years of calendar days
# = ~500 trading days = ~50,000 rows with 100 tickers).
# ==========================================================


def get_training_window():
    """
    Compute training date window.

    Uses TRAINING_LOOKBACK_DAYS env var (default 730 = 2 years).
    This is separate from INFERENCE_LOOKBACK_DAYS (400 days)
    used by the inference pipeline.
    """
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=TRAINING_LOOKBACK_DAYS)
    return (
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


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

    # Determine training window
    if start_date is None or end_date is None:
        # Try MarketTime first (respects market calendar)
        # Fall back to simple calendar window
        try:
            start_date, end_date = MarketTime.window_for("xgboost")
        except Exception:
            start_date, end_date = get_training_window()

    logger.info(
        "Training window | start=%s end=%s lookback_days=%d",
        start_date,
        end_date,
        TRAINING_LOOKBACK_DAYS,
    )

    # Load data and build features
    raw_df = load_training_data(start_date, end_date)

    # Walk-forward validation (produces metrics without data leakage)
    validator = WalkForwardValidator(trainer)
    metrics = validator.run(raw_df.copy())

    # Final model trained on full dataset
    final_df = build_target(raw_df)
    dataset_hash = compute_dataset_hash(final_df)
    final_model = trainer(raw_df)

    # Save model, metadata, pointer, optional baseline
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

    # Clean up old DB rows beyond retention window
    # Non-blocking — never stops training on failure
    cleanup_old_data()

    logger.info("Training completed successfully.")
    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MarketSentinel XGBoost Training Pipeline"
    )
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="Create a new drift baseline (first time only)",
    )
    parser.add_argument(
        "--promote-baseline",
        action="store_true",
        help="Update drift baseline from new training data (every retrain)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override training start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Override training end date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    main(
        start_date=args.start_date,
        end_date=args.end_date,
        create_baseline=args.create_baseline,
        promote_baseline=args.promote_baseline,
    )
