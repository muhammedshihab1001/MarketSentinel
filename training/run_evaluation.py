"""
MarketSentinel Institutional Evaluation Runner

Used by CI/CD pipeline.

Hard-fails on governance breach.
Explicit exit codes for CI.
"""

import sys
import os
import glob
import json
import joblib
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
from core.market.universe import MarketUniverse
from core.time.market_time import MarketTime
from training.evaluate import evaluate_xgboost


# =========================================================
# GOVERNANCE THRESHOLDS
# =========================================================

MIN_ROC_AUC = 0.50
MIN_SHARPE = 0.10
MIN_SPREAD = 0.0
MIN_SAMPLE_SIZE = 2000
FORWARD_DAYS = 5


# =========================================================
# CROSS-SECTIONAL FEATURE REBUILD (MUST MATCH TRAINING)
# =========================================================

def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(["date", "ticker"]).copy()

    base_cols = [
        "momentum_20",
        "return_lag5",
        "rsi",
        "volatility",
        "ema_ratio"
    ]

    for col in base_cols:

        if col not in df.columns:
            raise RuntimeError(f"Missing base feature: {col}")

        cs_mean = df.groupby("date")[col].transform("mean")
        cs_std = df.groupby("date")[col].transform("std")

        z = (df[col] - cs_mean) / (cs_std.replace(0, np.nan))
        z = z.clip(-5, 5)

        rank = df.groupby("date")[col].rank(pct=True)

        df[f"{col}_z"] = z.fillna(0.0)
        df[f"{col}_rank"] = rank.fillna(0.5)

    return df


# =========================================================
# TARGET REBUILD (MUST MATCH TRAINING)
# =========================================================

def apply_cross_sectional_target(df):

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
    df["target"] = df["target"].astype("int8")

    return df.reset_index(drop=True)


# =========================================================
# LOAD MODEL
# =========================================================

def load_latest_model():

    base_dir = os.path.join("artifacts", "xgboost")
    model_files = glob.glob(os.path.join(base_dir, "model_*.pkl"))

    if not model_files:
        raise RuntimeError("No trained model artifacts found.")

    latest = max(model_files, key=os.path.getmtime)
    timestamp = os.path.basename(latest).split("_")[1].split(".")[0]

    metadata_path = os.path.join(base_dir, f"metadata_{timestamp}.json")

    if not os.path.exists(metadata_path):
        raise RuntimeError("Metadata file missing.")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("schema_signature") != get_schema_signature():
        raise RuntimeError("Schema signature mismatch.")

    model = joblib.load(latest)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Artifact not classifier.")

    print(f"Loaded model version: {timestamp}")
    return model


# =========================================================
# DATASET BUILD
# =========================================================

def build_dataset():

    market_data = MarketDataService()
    store = FeatureStore()
    universe = MarketUniverse.get_universe()

    start_date, end_date = MarketTime.window_for("xgboost")

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
                training=False
            )

            if dataset is not None and not dataset.empty:
                datasets.append(dataset)

        except Exception:
            continue

    if not datasets:
        raise RuntimeError("No evaluation datasets built.")

    df = pd.concat(datasets, ignore_index=True)

    # 🔥 CRITICAL: must match training
    df = add_cross_sectional_features(df)
    df = apply_cross_sectional_target(df)

    feature_df = validate_feature_schema(df.loc[:, MODEL_FEATURES])
    df = df.loc[feature_df.index]

    if df["target"].nunique() < 2:
        raise RuntimeError("Labels collapsed.")

    if len(df) < MIN_SAMPLE_SIZE:
        raise RuntimeError("Dataset too small.")

    return df.reset_index(drop=True)


# =========================================================
# MAIN ENTRY (CI SAFE)
# =========================================================

def main() -> int:

    print("Starting CI evaluation...")

    init_env()

    model = load_latest_model()
    df = build_dataset()

    X = df.loc[:, MODEL_FEATURES]
    y = df["target"]
    forward_returns = df["forward_log_return"]
    dates = df["date"]

    probs = model.predict_proba(X)[:, 1]

    df_eval = df.copy()
    df_eval["prob"] = probs

    preds = np.full(len(df_eval), -1, dtype=int)

    for date, group in df_eval.groupby("date"):

        if len(group) < 5:
            continue

        long_threshold = group["prob"].quantile(0.80)
        short_threshold = group["prob"].quantile(0.20)

        long_mask = (df_eval["date"] == date) & (df_eval["prob"] >= long_threshold)
        short_mask = (df_eval["date"] == date) & (df_eval["prob"] <= short_threshold)

        preds[long_mask] = 1
        preds[short_mask] = 0

    active_mask = preds != -1

    if np.sum(active_mask) < 500:
        print("FAIL: Too few active signals.")
        return 1

    metrics = evaluate_xgboost(
        y_true=y[active_mask],
        y_pred=preds[active_mask],
        y_prob=probs[active_mask],
        forward_returns=forward_returns[active_mask],
        dates=dates[active_mask],
        enforce_thresholds=False
    )

    print("Evaluation metrics:", metrics)

    if metrics["roc_auc"] < MIN_ROC_AUC:
        print("FAIL: ROC AUC below gate.")
        return 1

    if metrics["sharpe"] < MIN_SHARPE:
        print("FAIL: Sharpe below gate.")
        return 1

    if metrics["long_short_spread"] < MIN_SPREAD:
        print("FAIL: Negative spread.")
        return 1

    print("CI evaluation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())