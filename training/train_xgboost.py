import os
import time
import json
import joblib
import logging
import random
import numpy as np
import pandas as pd

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature,
)
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse

from training.backtesting.walk_forward import WalkForwardValidator
from models.xgboost_model import build_xgboost_pipeline


logger = logging.getLogger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")

SEED = 42
MIN_TRAINING_ROWS = 1200
FORWARD_DAYS = 5


############################################################
# DETERMINISM
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# CROSS-SECTIONAL FEATURES
############################################################

def build_cross_sectional_features(df: pd.DataFrame):

    cross_cols = [
        "momentum_20",
        "return_lag5",
        "rsi",
        "volatility",
        "ema_ratio"
    ]

    for col in cross_cols:

        df[f"{col}_z"] = (
            df.groupby("date")[col]
            .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
        ).clip(-5, 5)

        df[f"{col}_rank"] = (
            df.groupby("date")[col]
            .transform(lambda x: x.rank(pct=True))
        )

    return df


############################################################
# CROSS-SECTIONAL TARGET
############################################################

def build_cross_sectional_target(df: pd.DataFrame):

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

    return df


############################################################
# LOAD DATA
############################################################

def load_training_data(start_date, end_date):

    market_data = MarketDataService()
    store = FeatureStore()
    universe = MarketUniverse.get_universe()

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

    df = build_cross_sectional_features(df)
    df = build_cross_sectional_target(df)

    # 🔥 Ensure feature alignment
    feature_df = validate_feature_schema(df.loc[:, MODEL_FEATURES])
    df = df.loc[feature_df.index]

    logger.info(
        "Target distribution | class_0=%s class_1=%s",
        (df["target"] == 0).sum(),
        (df["target"] == 1).sum()
    )

    if df["target"].nunique() < 2:
        raise RuntimeError("Training labels collapsed.")

    return df.reset_index(drop=True)


############################################################
# TRAINER
############################################################

def trainer(train_df):

    X = train_df.loc[:, MODEL_FEATURES]
    y = train_df["target"]

    pipeline = build_xgboost_pipeline(y)
    pipeline.fit(X, y)

    return pipeline


############################################################
# ARTIFACT EXPORT
############################################################

def export_artifacts(model, metrics):

    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = int(time.time())

    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
    joblib.dump(model, model_path)

    # Feature importance
    importance = model.export_feature_importance()

    importance_path = os.path.join(
        MODEL_DIR,
        f"feature_importance_{timestamp}.json"
    )

    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)

    # Metadata
    metadata = {
        "schema_signature": get_schema_signature(),
        "seed": SEED,
        "timestamp": timestamp,
        "metrics": metrics,
        "model_path": model_path
    }

    metadata_path = os.path.join(
        MODEL_DIR,
        f"metadata_{timestamp}.json"
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts exported successfully.")


############################################################
# MAIN
############################################################

def main(start_date=None, end_date=None):

    t0 = time.time()

    init_env()
    enforce_determinism()

    if not start_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    logger.info(
        "Training window | %s -> %s",
        start_date,
        end_date
    )

    df = load_training_data(start_date, end_date)

    validator = WalkForwardValidator(trainer)
    metrics = validator.run(df)

    logger.info("Walk-forward metrics: %s", metrics)

    final_model = trainer(df)

    export_artifacts(final_model, metrics)

    logger.info(
        "Training completed in %.2f minutes",
        (time.time() - t0) / 60
    )

    return metrics


if __name__ == "__main__":
    main()