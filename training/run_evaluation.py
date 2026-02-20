"""
Institutional XGBoost Evaluation Runner

Guarantees:
- Uses FeatureStore (canonical pipeline)
- Loads latest timestamped artifact
- Validates schema signature
- Rebuilds cross-sectional target (match training)
- Validates feature contract
- CI-safe & deterministic
"""

import os
import json
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
    get_schema_signature
)
from core.market.universe import MarketUniverse
from core.time.market_time import MarketTime


XGB_MIN_ACCURACY = 0.55
XGB_MIN_AUC = 0.60


############################################################
# ARTIFACT RESOLUTION (NEW)
############################################################

def load_latest_model():

    base_dir = os.path.join("artifacts", "xgboost")

    model_files = glob.glob(os.path.join(base_dir, "model_*.pkl"))

    if not model_files:
        raise RuntimeError("No trained model artifacts found.")

    latest = max(model_files, key=os.path.getmtime)

    timestamp = os.path.basename(latest).split("_")[1].split(".")[0]

    metadata_path = os.path.join(
        base_dir,
        f"metadata_{timestamp}.json"
    )

    if not os.path.exists(metadata_path):
        raise RuntimeError("Metadata file missing for latest model.")

    with open(metadata_path) as f:
        meta = json.load(f)

    if meta.get("schema_signature") != get_schema_signature():
        raise RuntimeError("Schema signature mismatch during evaluation.")

    model = joblib.load(latest)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded artifact is not a classifier.")

    print(f"Loaded model version: {timestamp}")

    return model


############################################################
# CROSS-SECTION TARGET (MATCH TRAINING EXACTLY)
############################################################

FORWARD_DAYS = 5


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


############################################################
# BUILD DATASET
############################################################

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
        raise RuntimeError("No valid datasets built.")

    df = pd.concat(datasets, ignore_index=True)

    df = apply_cross_sectional_target(df)

    # Validate feature schema strictly
    feature_df = validate_feature_schema(
        df.loc[:, MODEL_FEATURES]
    )

    df = df.loc[feature_df.index]

    return df.reset_index(drop=True)


############################################################
# EVALUATION
############################################################

def evaluate_xgb():

    model = load_latest_model()

    df = build_dataset()

    X = df.loc[:, MODEL_FEATURES]
    y = df["target"]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    accuracy = float(accuracy_score(y, preds))
    auc = float(roc_auc_score(y, probs))

    metrics = {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(auc, 4),
        "rows": len(df)
    }

    print("Evaluation metrics:", metrics)

    if accuracy < XGB_MIN_ACCURACY:
        raise RuntimeError(f"Accuracy below gate: {metrics}")

    if auc < XGB_MIN_AUC:
        raise RuntimeError(f"AUC below gate: {metrics}")

    return metrics


############################################################
# MAIN
############################################################

def main():

    init_env()

    metrics = evaluate_xgb()

    print("Evaluation passed.")
    print(metrics)


if __name__ == "__main__":
    main()