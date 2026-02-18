import os
import tempfile
import shutil
import time
import joblib
import pandas as pd
import numpy as np
import logging
import random
import gc

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature
)

from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from core.monitoring.drift_detector import DriftDetector

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
MIN_SURVIVING_RATIO = 0.20
MIN_SHARPE = 0.15
MAX_DRAWDOWN = -0.55
MIN_MODEL_BYTES = 50_000

PROMOTE_FLAG = os.getenv("ALLOW_MODEL_PROMOTION", "false").lower() == "true"
ALLOW_BASELINE_FLAG = os.getenv("ALLOW_DRIFT_BASELINE", "false").lower() == "true"

BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.58"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "0.42"))


############################################################
# DETERMINISM
############################################################

def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)


############################################################
# ATOMIC SAVE
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
# SAFE SCHEMA
############################################################

def safe_validate_schema(df):

    validated = validate_feature_schema(df)

    if isinstance(validated, tuple):
        validated = validated[0]

    return validated


############################################################
# 🔥 INSTITUTIONAL CROSS-SECTIONAL NORMALIZATION
############################################################

def cross_sectional_normalize(df):

    df = df.sort_values(["date", "ticker"]).copy()

    # ⭐ REQUIRE TRUE CROSS SECTION
    counts = df.groupby("date")["ticker"].transform("count")

    if (counts < 2).any():
        logger.warning(
            "Dropping single-asset dates before normalization."
        )
        df = df[counts >= 2]

    grouped = df.groupby("date")

    for col in MODEL_FEATURES:

        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std")

        # ⭐ CRITICAL FIX
        std = std.fillna(1.0).clip(lower=1e-6)

        df[col] = (df[col] - mean) / std

    return df.reset_index(drop=True)


############################################################
# DATA LOADER
############################################################

def load_training_data(start_date, end_date):

    market_data = MarketDataService()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    store = FeatureStore()

    universe = MarketUniverse.get_universe()

    datasets = []
    surviving = []

    for ticker in universe:

        try:

            price_df = market_data.get_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )

            sentiment_df = None

            try:

                news_df = news_fetcher.fetch(
                    f"{ticker} stock",
                    max_items=250
                )

                if news_df is not None and not news_df.empty:

                    scored = sentiment_analyzer.analyze_dataframe(news_df)

                    sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(
                        scored
                    )

            except Exception:
                logger.info(
                    "Sentiment disabled for %s — continuing.",
                    ticker
                )

            dataset = store.get_features(
                price_df,
                sentiment_df,
                ticker=ticker,
                training=True
            )

            datasets.append(dataset)
            surviving.append(ticker)

        except Exception as e:

            logger.warning(
                "Ticker rejected: %s | %s",
                ticker,
                str(e)
            )

    if not datasets:
        raise RuntimeError("All tickers failed — training aborted.")

    survival_ratio = len(surviving) / max(len(universe), 1)

    if survival_ratio < MIN_SURVIVING_RATIO:
        logger.warning(
            "Universe degraded — survival ratio %.2f",
            survival_ratio
        )

    # ⭐ SAFE CONCAT
    df = pd.concat(datasets, ignore_index=True, copy=False)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError("Training aborted — dataset too small.")

    ########################################################
    # NORMALIZE AFTER CONCAT
    ########################################################

    df = cross_sectional_normalize(df)

    ########################################################

    safe_validate_schema(df.loc[:, MODEL_FEATURES])

    features = df.loc[:, MODEL_FEATURES].to_numpy(dtype=np.float32)

    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite feature values detected.")

    if (np.var(features, axis=0) < 1e-7).any():
        logger.warning("Low feature variance detected.")

    if df["target"].nunique() < 2:
        raise RuntimeError("Target collapsed.")

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

    logger.info("Institutional XGBoost Training")

    if not start_date or not end_date:
        start_date, end_date = MarketTime.window_for("xgboost")

    df, dataset_hash = load_training_data(start_date, end_date)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ########################################################
    # TRAINER
    ########################################################

    def trainer(d):

        y = d["target"]

        model = build_xgboost_model(y)

        model.fit(d.loc[:, MODEL_FEATURES], y)

        return model

    ########################################################

    def signal(model, test):

        probs = model.predict_proba(
            test.loc[:, MODEL_FEATURES]
        )[:, 1]

        if np.std(probs) < 1e-6:
            logger.warning("Probability collapse detected.")

        return [
            "BUY" if p > BUY_THRESHOLD
            else "SELL" if p < SELL_THRESHOLD
            else "HOLD"
            for p in probs
        ]

    wf = WalkForwardValidator(
        model_trainer=trainer,
        signal_generator=signal
    )

    strategy_metrics = wf.run(df)

    gc.collect()

    if strategy_metrics["avg_sharpe"] < MIN_SHARPE:
        logger.warning("Sharpe below ideal threshold.")

    ########################################################
    # FINAL MODEL
    ########################################################

    final_model = build_final_xgboost_model(df["target"])

    final_model.fit(
        df.loc[:, MODEL_FEATURES],
        df["target"]
    )

    importance = dict(
        zip(
            MODEL_FEATURES,
            final_model.feature_importances_.tolist()
        )
    )

    save_model_atomic(final_model, TEMP_MODEL_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="xgboost_direction",
        metrics={**strategy_metrics},
        features=list(MODEL_FEATURES),
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        dataset_rows=len(df),
        metadata_type="training_manifest_v1",
        extra_fields={
            "feature_importance": importance,
            "buy_threshold": BUY_THRESHOLD,
            "sell_threshold": SELL_THRESHOLD,
            "time_snapshot": MarketTime.snapshot_for("xgboost")
        }
    )

    MetadataManager.save_metadata(
        metadata,
        TEMP_METADATA_PATH
    )

    version = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    if ALLOW_BASELINE_FLAG and PROMOTE_FLAG:

        drift = DriftDetector()

        try:
            drift.create_baseline(
                df,
                dataset_hash=dataset_hash,
                allow_overwrite=False
            )
        except RuntimeError:
            pass

    if PROMOTE_FLAG:
        ModelRegistry.promote_to_production(
            MODEL_DIR,
            version
        )
        logger.info("Model promoted → version=%s", version)

    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    logger.info(
        "Training summary | version=%s rows=%s sharpe=%.3f",
        version,
        len(df),
        strategy_metrics["avg_sharpe"]
    )

    logger.info(
        "Total training time: %.2f minutes",
        (time.time() - t0) / 60
    )

    return strategy_metrics


if __name__ == "__main__":
    main()
