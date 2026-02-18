import os
import tempfile
import shutil
import time
import joblib
import pandas as pd
import numpy as np
import logging
import random

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
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

TEMP_DIR = tempfile.mkdtemp(prefix="xgb_train_")
TEMP_MODEL_PATH = os.path.join(TEMP_DIR, "model.pkl")
TEMP_METADATA_PATH = os.path.join(TEMP_DIR, "metadata.json")

SEED = 42
MIN_TRAINING_ROWS = 1200
MIN_MODEL_BYTES = 50_000


############################################################
# DETERMINISM
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# SAVE MODEL SAFELY
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
# CROSS SECTION NORMALIZATION
############################################################

def cross_sectional_normalize(df):

    df = df.sort_values(["date", "ticker"]).copy()

    counts = df.groupby("date")["ticker"].transform("count")
    df = df[counts >= 2]

    grouped = df.groupby("date")

    for col in MODEL_FEATURES:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").fillna(1).clip(lower=1e-6)
        df[col] = (df[col] - mean) / std

    return df.reset_index(drop=True)


############################################################
# DATA LOADER WITH ALPHA PRUNING
############################################################

def load_training_data(start_date, end_date):

    logger.info("Loading institutional dataset...")

    market_data = MarketDataService()
    store = FeatureStore()

    universe = MarketUniverse.get_universe()

    datasets = []
    ticker_scores = {}

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

            # Simple information coefficient proxy
            ic = abs(dataset["target"].corr(dataset["return_lag1"]))
            ticker_scores[ticker] = ic if np.isfinite(ic) else 0

            datasets.append(dataset)

            logger.info("Ticker accepted → %s | IC=%.3f", ticker, ic)

        except Exception as e:
            logger.warning("Ticker rejected → %s | %s", ticker, str(e))

    if not datasets:
        raise RuntimeError("All tickers failed.")

    ###################################################
    # Alpha pruning
    ###################################################

    ranked = sorted(
        ticker_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    keep_n = max(6, int(len(ranked) * 0.6))
    keep = {t for t, _ in ranked[:keep_n]}

    logger.warning("Alpha pruning active → keeping %s tickers", len(keep))

    filtered = [
        d for d in datasets
        if d["ticker"].iloc[0] in keep
    ]

    df = pd.concat(filtered, ignore_index=True)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError("Dataset too small.")

    df = cross_sectional_normalize(df)

    validate_feature_schema(df.loc[:, MODEL_FEATURES])

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

    logger.info("Final dataset rows=%s", len(df))

    return df, dataset_hash


############################################################
# MAIN TRAINING PIPELINE
############################################################

def main(start_date=None, end_date=None):

    t0 = time.time()

    init_env()
    enforce_determinism()

    logger.info("===== INSTITUTIONAL XGBOOST TRAINING =====")

    if not start_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    df, dataset_hash = load_training_data(start_date, end_date)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ########################################################
    # MODEL TRAINER
    ########################################################

    def trainer(d):

        y = d["target"]
        model = build_xgboost_model(y)
        model.fit(d.loc[:, MODEL_FEATURES], y)

        return model

    ########################################################
    # PROBABILITY SIGNAL GENERATOR (WF COMPATIBLE)
    ########################################################

    def wf_signal_generator(model, test_df):

        probs = model.predict_proba(
            test_df.loc[:, MODEL_FEATURES]
        )[:, 1]

        signals = [
            "BUY" if p > 0.58
            else "SELL" if p < 0.42
            else "HOLD"
            for p in probs
        ]

        return signals, probs

    ########################################################
    # WALK FORWARD
    ########################################################

    wf = WalkForwardValidator(
        model_trainer=trainer,
        signal_generator=wf_signal_generator
    )

    logger.info("Running institutional walk-forward...")

    metrics = wf.run(df)

    logger.info(
        "WF complete | Sharpe=%.3f Return=%.3f",
        metrics["avg_sharpe"],
        metrics["avg_strategy_return"]
    )

    ########################################################
    # FINAL PRODUCTION MODEL
    ########################################################

    final_model = build_final_xgboost_model(df["target"])

    final_model.fit(
        df.loc[:, MODEL_FEATURES],
        df["target"]
    )

    size = save_model_atomic(final_model, TEMP_MODEL_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics=metrics,
        features=list(MODEL_FEATURES),
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        dataset_rows=len(df),
        metadata_type="training_manifest_v1"
    )

    MetadataManager.save_metadata(metadata, TEMP_METADATA_PATH)

    version = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    logger.info("MODEL VERSION → %s", version)
    logger.info("TOTAL TIME → %.2f minutes", (time.time() - t0) / 60)

    return metrics


if __name__ == "__main__":
    main()
