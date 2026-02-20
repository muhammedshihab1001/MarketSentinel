import os
import time
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
)
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse

from training.backtesting.walk_forward import WalkForwardValidator
from models.xgboost_model import build_xgboost_pipeline


logger = logging.getLogger("marketsentinel.train_xgb")

MODEL_DIR = os.path.abspath("artifacts/xgboost")

SEED = 42
MIN_TRAINING_ROWS = 1200


############################################################
# DETERMINISM
############################################################

def enforce_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# CROSS-SECTIONAL TARGET (INSTITUTIONAL FIXED VERSION)
############################################################

def apply_cross_sectional_target(df):

    df = df.sort_values(["date", "ticker"]).copy()

    labeled_frames = []

    for date, group in df.groupby("date"):

        if len(group) < 6:
            continue

        group = group.copy()

        # risk-adjusted return
        safe_vol = group["volatility"].clip(lower=1e-4)
        group["risk_adj"] = group["forward_return"] / safe_vol

        # require dispersion
        if group["risk_adj"].std() < 0.05:
            continue

        lower_q = group["risk_adj"].quantile(0.3)
        upper_q = group["risk_adj"].quantile(0.7)

        group["target"] = np.nan
        group.loc[group["risk_adj"] <= lower_q, "target"] = 0
        group.loc[group["risk_adj"] >= upper_q, "target"] = 1

        labeled = group.dropna(subset=["target"])

        if labeled["target"].nunique() == 2:
            labeled_frames.append(labeled)

    if not labeled_frames:
        raise RuntimeError("Cross-sectional labeling failed.")

    df = pd.concat(labeled_frames, ignore_index=True)
    df["target"] = df["target"].astype("int8")

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

    validate_feature_schema(df.loc[:, MODEL_FEATURES])

    # 🔥 APPLY TARGET HERE
    df = apply_cross_sectional_target(df)

    logger.info(
        "Target distribution | class_0=%s class_1=%s",
        (df["target"] == 0).sum(),
        (df["target"] == 1).sum()
    )

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

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")

    joblib.dump(final_model, model_path)

    logger.info(
        "Training completed in %.2f minutes",
        (time.time() - t0) / 60
    )

    return metrics


if __name__ == "__main__":
    main()