import os
import time
import joblib
import logging
import random
import numpy as np
import pandas as pd

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
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
# SENTIMENT AGGREGATION
############################################################

def build_sentiment_frame(ticker, start_date, end_date):

    news_fetcher = NewsFetcher()
    analyzer = SentimentAnalyzer()

    try:
        news_df = news_fetcher.fetch(
            query=ticker,
            max_items=150
        )

        if news_df.empty:
            return None

        analyzed = analyzer.analyze_dataframe(news_df)

        analyzed["date"] = pd.to_datetime(
            analyzed["published_at"]
        ).dt.normalize()

        grouped = analyzed.groupby("date").agg(
            avg_sentiment=("score", "mean"),
            sentiment_std=("score", "std"),
            news_count=("score", "count")
        ).reset_index()

        grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)

        return grouped

    except Exception as e:
        logger.warning("Sentiment build failed for %s: %s", ticker, str(e))
        return None


############################################################
# CROSS-SECTION TARGET
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

    if df["target"].nunique() < 2:
        raise RuntimeError("Label collapse detected.")

    return df.reset_index(drop=True)


############################################################
# LOAD DATA
############################################################

def load_training_data(start_date, end_date):

    market_data = MarketDataService()
    store = FeatureStore()
    universe = MarketUniverse.get_universe()

    datasets = []
    failed = []

    for ticker in universe:

        try:
            logger.info("Building features for %s", ticker)

            price_df = market_data.get_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )

            # 🔥 SENTIMENT BUILD
            sentiment_df = build_sentiment_frame(
                ticker,
                start_date,
                end_date
            )

            dataset = store.get_features(
                price_df,
                sentiment_df=sentiment_df,
                ticker=ticker,
                training=True
            )

            if dataset is None or dataset.empty:
                raise RuntimeError("Empty feature dataset.")

            datasets.append(dataset)

        except Exception as e:
            logger.error("Ticker failed: %s | %s", ticker, str(e))
            failed.append(ticker)

    if not datasets:
        raise RuntimeError(
            f"All tickers failed. Failed tickers: {failed}"
        )

    df = pd.concat(datasets, ignore_index=True)

    logger.info("Raw dataset rows: %s", len(df))

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError(
            f"Dataset too small after feature build. Rows={len(df)}"
        )

    ########################################################
    # 🔐 SCHEMA VALIDATION
    ########################################################

    validate_feature_schema(df.loc[:, MODEL_FEATURES])

    ########################################################

    df = apply_cross_sectional_target(df)

    logger.info("Final labeled dataset rows: %s", len(df))

    return df


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