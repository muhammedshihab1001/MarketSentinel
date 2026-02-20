"""
Institutional XGBoost Evaluation Runner

Guarantees:
- Uses FeatureStore (canonical pipeline)
- Uses trained sklearn Pipeline model
- Rebuilds cross-sectional target
- Prevents training/inference drift
- CI-safe
"""

import datetime
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import MODEL_FEATURES
from core.market.universe import MarketUniverse
from core.time.market_time import MarketTime


XGB_MIN_ACCURACY = 0.55
XGB_MIN_AUC = 0.60


############################################################
# CROSS-SECTION TARGET (MATCH TRAINING)
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
        raise RuntimeError("No valid labels generated.")

    df = pd.concat(labeled, ignore_index=True)
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
                training=True
            )

            datasets.append(dataset)

        except Exception:
            continue

    if not datasets:
        raise RuntimeError("No valid datasets built.")

    df = pd.concat(datasets, ignore_index=True)

    df = apply_cross_sectional_target(df)

    return df


############################################################
# EVALUATION
############################################################

def evaluate_xgb():

    model_path = os.path.join("artifacts", "xgboost", "model.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError("Trained XGBoost model not found.")

    model = joblib.load(model_path)

    df = build_dataset()

    X = df.loc[:, MODEL_FEATURES]
    y = df["target"]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    accuracy = float(accuracy_score(y, preds))
    auc = float(roc_auc_score(y, probs))

    metrics = {
        "accuracy": accuracy,
        "roc_auc": auc
    }

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
