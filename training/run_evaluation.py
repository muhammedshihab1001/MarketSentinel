"""
MarketSentinel Institutional Evaluation Runner v3
Production-grade CI validator.
Strict governance enforcement.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_store import FeatureStore
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
)
from core.market.universe import MarketUniverse
from core.time.market_time import MarketTime
from core.monitoring.drift_detector import DriftDetector
from app.inference.model_loader import ModelLoader
from training.evaluate import evaluate_xgboost


# =========================================================
# GOVERNANCE THRESHOLDS
# =========================================================

MIN_ROC_AUC = 0.50
MIN_SHARPE = 0.10
MIN_SPREAD = 0.0
MIN_SAMPLE_SIZE = 2000
MIN_ACTIVE_SIGNALS = 500

MAX_SHARPE_DEGRADATION = 0.10
MAX_ALLOWED_DRIFT = 0.35

FORWARD_DAYS = 5
LONG_PERCENTILE = 0.70
SHORT_PERCENTILE = 0.30


# =========================================================
# TARGET REBUILD
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
    df.loc[df["alpha_rank_pct"] >= LONG_PERCENTILE, "target"] = 1
    df.loc[df["alpha_rank_pct"] <= SHORT_PERCENTILE, "target"] = 0

    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype("int8")

    return df.reset_index(drop=True)


# =========================================================
# FEATURE ALIGNMENT
# =========================================================

def align_to_model_features(df: pd.DataFrame) -> pd.DataFrame:

    missing = [f for f in MODEL_FEATURES if f not in df.columns]

    if missing:
        print(f"WARNING: Missing features detected → {missing}")
        for col in missing:
            df[col] = 0.0

    df = df.loc[:, MODEL_FEATURES]
    return df


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
    df = apply_cross_sectional_target(df)

    X_aligned = align_to_model_features(df)

    validate_feature_schema(
        X_aligned,
        mode="strict_contract"
    )

    df = df.loc[X_aligned.index]

    if df["target"].nunique() < 2:
        raise RuntimeError("Labels collapsed.")

    if len(df) < MIN_SAMPLE_SIZE:
        raise RuntimeError("Dataset too small.")

    return df.reset_index(drop=True)


# =========================================================
# BASELINE COMPARISON (ROBUST)
# =========================================================

def compare_to_baseline(metrics):

    baseline_path = Path("artifacts/xgboost/baseline_contract.json")

    if not baseline_path.exists():
        print("No baseline contract found.")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_metrics = baseline.get("metrics", {})

    baseline_sharpe = baseline_metrics.get("avg_sharpe")

    if baseline_sharpe is None:
        return

    if metrics["sharpe"] < baseline_sharpe - MAX_SHARPE_DEGRADATION:
        raise RuntimeError(
            f"Sharpe degraded vs baseline. "
            f"Current={metrics['sharpe']:.4f} "
            f"Baseline={baseline_sharpe:.4f}"
        )


# =========================================================
# MAIN
# =========================================================

def main() -> int:

    print("Starting Institutional CI evaluation...")
    init_env()

    loader = ModelLoader()
    model = loader.xgb

    print(f"Evaluating model version: {loader.xgb_version}")
    print(f"Schema signature: {loader.schema_signature[:12]}")

    df = build_dataset()

    X = align_to_model_features(df)
    y = df["target"]
    forward_returns = df["forward_log_return"]
    dates = df["date"]

    # Drift validation
    drift_detector = DriftDetector()
    drift_score = drift_detector.compute_drift(X)

    print(f"Drift score: {drift_score:.4f}")

    if drift_score > MAX_ALLOWED_DRIFT:
        raise RuntimeError("Drift exceeded governance limit.")

    probs = model.predict_proba(X)[:, 1]

    df_eval = df.copy()
    df_eval["prob"] = probs

    preds = np.full(len(df_eval), -1, dtype=int)

    for date, group in df_eval.groupby("date"):

        if len(group) < 5:
            continue

        long_threshold = group["prob"].quantile(LONG_PERCENTILE)
        short_threshold = group["prob"].quantile(SHORT_PERCENTILE)

        long_mask = (
            (df_eval["date"] == date) &
            (df_eval["prob"] >= long_threshold)
        )

        short_mask = (
            (df_eval["date"] == date) &
            (df_eval["prob"] <= short_threshold)
        )

        preds[long_mask] = 1
        preds[short_mask] = 0

    active_mask = preds != -1

    if np.sum(active_mask) < MIN_ACTIVE_SIGNALS:
        raise RuntimeError("Too few active signals.")

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
        raise RuntimeError("ROC AUC below threshold.")

    if metrics["sharpe"] < MIN_SHARPE:
        raise RuntimeError("Sharpe below threshold.")

    if metrics["long_short_spread"] < MIN_SPREAD:
        raise RuntimeError("Negative long-short spread.")

    compare_to_baseline(metrics)

    print("CI evaluation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())