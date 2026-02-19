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
# CROSS SECTION LABELING (MARKET NEUTRAL)
############################################################

def apply_cross_sectional_target(df):

    df = df.sort_values(["date", "ticker"]).copy()

    # ----------------------------------------------------
    # 1️⃣ Extract benchmark return
    # ----------------------------------------------------
    benchmark = df[df["ticker"] == BENCHMARK_TICKER][
        ["date", "forward_return"]
    ].rename(columns={"forward_return": "benchmark_return"})

    df = df.merge(benchmark, on="date", how="left")

    df["benchmark_return"] = df["benchmark_return"].fillna(0.0)

    # ----------------------------------------------------
    # 2️⃣ Market-neutral alpha
    # ----------------------------------------------------
    df["alpha"] = df["forward_return"] - df["benchmark_return"]

    safe_vol = df["volatility"].clip(lower=1e-4)
    df["risk_adj"] = (df["alpha"] / safe_vol).clip(-5, 5)

    labeled = []

    for date, group in df.groupby("date"):

        if len(group) < 5:
            continue

        upper = group["risk_adj"].quantile(0.80)
        lower = group["risk_adj"].quantile(0.20)

        group = group.copy()

        group["target"] = np.where(
            group["risk_adj"] >= upper, 1,
            np.where(group["risk_adj"] <= lower, 0, np.nan)
        )

        labeled.append(group)

    df = pd.concat(labeled, ignore_index=True)

    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype("int8")

    logger.info(
        "Class balance | pos=%s neg=%s",
        int(np.sum(df["target"] == 1)),
        int(np.sum(df["target"] == 0))
    )

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

            validate_feature_schema(dataset.loc[:, MODEL_FEATURES])
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

    if df["target"].nunique() < 2:
        raise RuntimeError("Label collapse after labeling.")

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


############################################################
# MAIN
############################################################

def main(start_date=None, end_date=None):

    t0 = time.time()

    init_env()
    enforce_determinism()

    logger.info("INSTITUTIONAL XGBOOST TRAINING STARTED")

    if not start_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    df, dataset_hash = load_training_data(start_date, end_date)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ########################################################
    # TRAINER
    ########################################################

    def trainer(d):

        d = d.sort_values("date")

        X = d.loc[:, MODEL_FEATURES]
        y = d["target"]

        split_index = int(len(d) * 0.85)

        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]

        X_val = X.iloc[split_index:]
        y_val = y.iloc[split_index:]

        model = build_xgboost_model(y_train)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        calibrated = CalibratedClassifierCV(
            model,
            method="isotonic",
            cv="prefit"
        )

        calibrated.fit(X_val, y_val)

        return calibrated

    ########################################################
    # FIXED RANK-BASED SIGNAL GENERATOR
    ########################################################

    def signal_generator(model, test_df):

        probs = model.predict_proba(
            test_df.loc[:, MODEL_FEATURES]
        )[:, 1]

        df_local = test_df.copy()
        df_local["prob"] = probs
        df_local["signal"] = "HOLD"

        for date, group in df_local.groupby("date"):

            k = max(1, int(len(group) * TOP_K_PERCENT))

            sorted_group = group.sort_values("prob", ascending=False)

            long_idx = sorted_group.head(k).index
            short_idx = sorted_group.tail(k).index

            df_local.loc[long_idx, "signal"] = "BUY"
            df_local.loc[short_idx, "signal"] = "SELL"

        return df_local["signal"].tolist(), probs

    ########################################################
    # WALK FORWARD
    ########################################################

    wf = WalkForwardValidator(
        model_trainer=trainer,
        signal_generator=signal_generator
    )

    metrics = wf.run(df)

    logger.info(
        "WF complete | Sharpe=%.3f Return=%.3f",
        metrics["avg_sharpe"],
        metrics["avg_strategy_return"]
    )

    ########################################################
    # FINAL MODEL
    ########################################################

    final_model = build_final_xgboost_model(df["target"])
    final_model.fit(df.loc[:, MODEL_FEATURES], df["target"])

    importances = final_model.feature_importances_
    for name, val in zip(MODEL_FEATURES, importances):
        logger.info("Feature importance | %s = %.6f", name, float(val))

    with tempfile.TemporaryDirectory(prefix="xgb_train_") as tmpdir:

        model_path = os.path.join(tmpdir, "model.pkl")
        metadata_path = os.path.join(tmpdir, "metadata.json")

        save_model_atomic(final_model, model_path)

        clean_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if not isinstance(v, list)
        }

        metadata = MetadataManager.create_metadata(
            model_name="xgboost_direction",
            metrics=clean_metrics,
            features=list(MODEL_FEATURES),
            training_start=start_date,
            training_end=end_date,
            dataset_hash=dataset_hash,
            dataset_rows=len(df),
            metadata_type="training_manifest_v1"
        )

        MetadataManager.save_metadata(metadata, metadata_path)

        version = ModelRegistry.register_model(
            MODEL_DIR,
            model_path,
            metadata_path
        )

    logger.info("MODEL VERSION %s", version)
    logger.info("TOTAL TIME %.2f minutes", (time.time() - t0) / 60)

    return metrics


if __name__ == "__main__":
    main()
