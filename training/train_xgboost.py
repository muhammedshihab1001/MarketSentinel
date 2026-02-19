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
# TRAINING DATA LOADER
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
        raise RuntimeError("Dataset too small.")

    logger.info("Final dataset rows=%s", len(df))

    return df.reset_index(drop=True)


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

    df = load_training_data(start_date, end_date)

    ############################################################
    # MODEL TRAINER FOR WALK FORWARD
    ############################################################

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

    ############################################################
    # WALK FORWARD
    ############################################################

    wf = WalkForwardValidator(
        model_trainer=trainer
    )

    metrics = wf.run(df)

    logger.info(
        "WF complete | Sharpe=%.3f Return=%.3f",
        metrics["avg_sharpe"],
        metrics["avg_strategy_return"]
    )

    ############################################################
    # FINAL PRODUCTION MODEL
    ############################################################

    final_model = build_final_xgboost_model(df["target"])

    final_model.fit(
        df.loc[:, MODEL_FEATURES],
        df["target"]
    )

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
            dataset_hash=get_schema_signature(),
            dataset_rows=len(df),
            metadata_type="training_manifest_v2"
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
