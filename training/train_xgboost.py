import os
import time
import joblib
import logging
import random
import hashlib
import numpy as np
import pandas as pd

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


logger = logging.getLogger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")

SEED = 42
MIN_TRAINING_ROWS = 1200
MIN_MODEL_BYTES = 50_000

MIN_OOS_SHARPE = 0.60
MAX_ALLOWED_DRAWDOWN = -0.45


############################################################
# DETERMINISM
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# SAFE SAVE
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
# CROSS-SECTION LABELING
############################################################

def apply_cross_sectional_target(df):

    df = df.sort_values(["date", "ticker"]).copy()

    df["alpha"] = df["forward_return"]
    safe_vol = df["volatility"].clip(lower=1e-4)
    df["risk_adj"] = (df["alpha"] / safe_vol).clip(-5, 5)

    labeled = []

    for date, group in df.groupby("date"):

        if len(group) < 6:
            continue

        dispersion = group["risk_adj"].std()

        if not np.isfinite(dispersion) or dispersion < 0.10:
            continue

        upper = group["risk_adj"].quantile(0.80)
        lower = group["risk_adj"].quantile(0.20)

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

    if df["target"].nunique() != 2:
        raise RuntimeError("Label collapse detected.")

    return df.reset_index(drop=True)


############################################################
# CROSS-SECTION NORMALIZATION
############################################################

def cross_sectional_normalize(df):

    df = df.sort_values(["date", "ticker"]).copy()
    grouped = df.groupby("date")

    stats = {}

    for col in MODEL_FEATURES:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").fillna(1).clip(lower=1e-6)
        df[col] = (df[col] - mean) / std

    return df.reset_index(drop=True)


############################################################
# LOAD DATA
############################################################

def load_training_data(start_date, end_date):

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

    df = apply_cross_sectional_target(df)
    df = cross_sectional_normalize(df)

    return df


############################################################
# WALK-FORWARD VALIDATION
############################################################

def run_walk_forward(df):

    validator = WalkForwardValidator()

    X = df.loc[:, MODEL_FEATURES]
    y = df["target"]

    model = build_xgboost_model(y)

    results = validator.validate(
        model=model,
        features=X,
        target=y,
        full_dataframe=df
    )

    sharpe = results.get("sharpe_ratio", 0.0)
    drawdown = results.get("max_drawdown", 0.0)

    logger.info("Walk-forward Sharpe=%.4f", sharpe)
    logger.info("Walk-forward MaxDD=%.4f", drawdown)

    if sharpe < MIN_OOS_SHARPE:
        raise RuntimeError("Model rejected: insufficient OOS Sharpe.")

    if drawdown < MAX_ALLOWED_DRAWDOWN:
        raise RuntimeError("Model rejected: excessive drawdown.")

    return results


############################################################
# MAIN
############################################################

def main(start_date=None, end_date=None):

    t0 = time.time()

    init_env()
    enforce_determinism()

    if not start_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    df = load_training_data(start_date, end_date)

    wf_results = run_walk_forward(df)

    X = df.loc[:, MODEL_FEATURES]
    y = df["target"]

    base_model = build_final_xgboost_model(y)
    calibrated = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=3
    )

    calibrated.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    save_model_atomic(calibrated, model_path)

    dataset_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df).values
    ).hexdigest()

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_cross_sectional",
        metrics=wf_results,
        features=MODEL_FEATURES,
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        dataset_rows=len(df),
        metadata_type="cross_section_manifest_v1",
        extra_fields={
            "schema_signature": get_schema_signature(),
            "universe_hash": MarketUniverse.fingerprint()
        }
    )

    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    MetadataManager.save_metadata(metadata, metadata_path)

    version = ModelRegistry.register_model(
        MODEL_DIR,
        model_path,
        metadata_path
    )

    logger.info("Model registered -> %s", version)
    logger.info("Training completed in %.2f minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
