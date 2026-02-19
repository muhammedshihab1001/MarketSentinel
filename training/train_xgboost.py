import os
import tempfile
import time
import joblib
import pandas as pd
import numpy as np
import logging
import random

from sklearn.calibration import CalibratedClassifierCV

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import (
    MODEL_FEATURES,
    get_schema_signature
)

from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry

from training.backtesting.walk_forward import WalkForwardValidator
from models.xgboost_model import (
    build_xgboost_model,
    build_final_xgboost_model
)

from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse


logger = logging.getLogger(__name__)

MODEL_DIR = os.path.abspath("artifacts/xgboost")

SEED = 42
MIN_TRAINING_ROWS = 1200
MIN_MODEL_BYTES = 50_000

TOP_K_PERCENT = 0.20
BENCHMARK_TICKER = "SPY"


############################################################
# DETERMINISM
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# SAFE MODEL SAVE
############################################################

def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = f"{path}.tmp"
    joblib.dump(model, tmp_path)
    os.replace(tmp_path, path)

    size = os.path.getsize(path)

    if size < MIN_MODEL_BYTES:
        raise RuntimeError("Model artifact suspiciously small.")

    joblib.load(path)

    return size


############################################################
# INSTITUTIONAL CROSS-SECTIONAL LABELING
############################################################

def apply_cross_sectional_target(df):

    df = df.sort_values(["date", "ticker"]).copy()

    # ---------------------------------------------
    # Benchmark merge (optional but preferred)
    # ---------------------------------------------
    if BENCHMARK_TICKER in df["ticker"].unique():

        benchmark = df[df["ticker"] == BENCHMARK_TICKER][
            ["date", "forward_return"]
        ].rename(columns={"forward_return": "benchmark_return"})

        df = df.merge(benchmark, on="date", how="left")
        df["benchmark_return"] = df["benchmark_return"].fillna(0.0)

    else:
        logger.warning("Benchmark missing — fallback to raw forward_return.")
        df["benchmark_return"] = 0.0

    df["alpha"] = df["forward_return"] - df["benchmark_return"]

    # Risk adjust
    safe_vol = df["volatility"].clip(lower=1e-4)
    df["risk_adj"] = (df["alpha"] / safe_vol).clip(-5, 5)

    labeled = []

    for date, group in df.groupby("date"):

        if len(group) < 6:
            continue

        dispersion = group["risk_adj"].std()

        # Skip flat days
        if not np.isfinite(dispersion) or dispersion < 0.10:
            continue

        upper_q = 0.80
        lower_q = 0.20

        upper = group["risk_adj"].quantile(upper_q)
        lower = group["risk_adj"].quantile(lower_q)

        group = group.copy()

        group["target"] = np.where(
            group["risk_adj"] >= upper, 1,
            np.where(group["risk_adj"] <= lower, 0, np.nan)
        )

        labeled.append(group)

    if not labeled:
        raise RuntimeError("No valid cross-sectional labels generated.")

    df = pd.concat(labeled, ignore_index=True)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype("int8")

    pos = int((df["target"] == 1).sum())
    neg = int((df["target"] == 0).sum())

    logger.info("Class balance | pos=%s neg=%s", pos, neg)

    if pos == 0 or neg == 0:
        raise RuntimeError("Label collapse detected.")

    return df.reset_index(drop=True)


############################################################
# CROSS SECTION NORMALIZATION
############################################################

def cross_sectional_normalize(df):

    df = df.sort_values(["date", "ticker"]).copy()
    grouped = df.groupby("date")

    for col in MODEL_FEATURES:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").fillna(1).clip(lower=1e-6)
        df[col] = (df[col] - mean) / std

    return df.reset_index(drop=True)


############################################################
# DATA LOADER
############################################################

def load_training_data(start_date, end_date):

    logger.info("Loading institutional dataset")

    market_data = MarketDataService()
    store = FeatureStore()
    universe = MarketUniverse.get_universe()

    datasets = []

    for ticker in universe:

        try:

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

            datasets.append(dataset)

        except Exception as e:

            logger.warning("Ticker rejected %s | %s", ticker, str(e))

    if not datasets:
        raise RuntimeError("All tickers failed.")

    df = pd.concat(datasets, ignore_index=True)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError("Dataset too small before labeling.")

    logger.info("Raw dataset rows=%s", len(df))

    df = apply_cross_sectional_target(df)

    logger.info("Post-label rows=%s", len(df))

    df = cross_sectional_normalize(df)

    logger.info("Final normalized rows=%s", len(df))

    hash_df = df[
        ["ticker", "date", "target", *MODEL_FEATURES]
    ].sort_values(["date", "ticker"]).reset_index(drop=True)

    dataset_hash = MetadataManager.hash_list([
        MetadataManager.fingerprint_dataset(hash_df),
        get_schema_signature(),
        list(MODEL_FEATURES),
        start_date,
        end_date
    ])

    return df, dataset_hash

if __name__ == "__main__":
    print("TRAINING ENTRYPOINT TRIGGERED")
    main()
