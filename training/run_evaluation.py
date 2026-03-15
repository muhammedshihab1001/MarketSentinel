"""
MarketSentinel Institutional Evaluation Runner v4
Aligned with Walk-Forward Portfolio Validation
Hybrid-Ready Governance Layer
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from core.config.env_loader import init_env
from core.data.market_data_service import MarketDataService
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import (
    MODEL_FEATURES,
    validate_feature_schema,
)
from core.market.universe import MarketUniverse
from core.time.market_time import MarketTime
from core.monitoring.drift_detector import DriftDetector
from app.inference.model_loader import ModelLoader
from training.backtesting.walk_forward import WalkForwardValidator
from training.train_xgboost import trainer as retrain_model


# =========================================================
# GOVERNANCE THRESHOLDS
# =========================================================

MIN_SHARPE = 0.10
MAX_DRAWDOWN = -0.60
MIN_WINDOWS = 5
MAX_SHARPE_DEGRADATION = 0.10
MAX_ALLOWED_DRIFT = 0.35


# =========================================================
# DATASET BUILD (FULL FEATURE PIPELINE)
# =========================================================

def build_dataset():
    """
    Fetch price data for all universe tickers, concatenate,
    then run the full feature pipeline ONCE on the combined
    dataset so cross-sectional features compute correctly.
    """

    market_data = MarketDataService()
    universe = MarketUniverse.get_universe()

    start_date, end_date = MarketTime.window_for("xgboost")

    # ── Step 1: Fetch raw price data per ticker ──────────────────────────
    price_frames = []
    fetch_failures = []

    for ticker in universe:
        try:
            price_df = market_data.get_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            if price_df is not None and not price_df.empty:
                price_frames.append(price_df)
        except Exception as exc:
            fetch_failures.append(ticker)
            continue

    if not price_frames:
        raise RuntimeError("No evaluation price data fetched.")

    if fetch_failures:
        print(f"Price fetch failed for {len(fetch_failures)} tickers: {fetch_failures}")

    # ── Step 2: Concatenate and run feature pipeline once ────────────────
    combined_prices = pd.concat(price_frames, ignore_index=True)

    df = FeatureEngineer.build_feature_pipeline(
        combined_prices, training=False
    )

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if len(df) < 2000:
        raise RuntimeError(
            f"Dataset too small for evaluation: {len(df)} rows, need 2000."
        )

    print(
        f"Evaluation dataset built | rows={len(df)} "
        f"tickers={df['ticker'].nunique()} dates={df['date'].nunique()}"
    )

    return df


# =========================================================
# BASELINE COMPARISON
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

    if metrics["avg_sharpe"] < baseline_sharpe - MAX_SHARPE_DEGRADATION:
        raise RuntimeError(
            f"Sharpe degraded vs baseline. "
            f"Current={metrics['avg_sharpe']:.4f} "
            f"Baseline={baseline_sharpe:.4f}"
        )


# =========================================================
# MAIN
# =========================================================

def main() -> int:

    print("Starting Institutional Walk-Forward CI Evaluation...")
    init_env()

    loader = ModelLoader()

    print(f"Evaluating model version: {loader.xgb_version}")
    print(f"Schema signature: {loader.schema_signature[:12]}")

    # Build dataset
    df = build_dataset()

    # Validate feature schema strictly
    X = validate_feature_schema(
        df.loc[:, MODEL_FEATURES],
        mode="strict_contract"
    )

    # Drift check
    drift_detector = DriftDetector()
    drift_score = drift_detector.compute_drift(X)

    print(f"Drift score: {drift_score:.4f}")

    if drift_score > MAX_ALLOWED_DRIFT:
        raise RuntimeError("Drift exceeded governance limit.")

    # Walk-forward validation — retrain_model retrains per fold
    # so each window gets a fresh model trained only on in-sample data
    validator = WalkForwardValidator(
        model_trainer=retrain_model,
    )

    metrics = validator.run(df.copy())

    print("Walk-forward metrics:", metrics)

    # Governance enforcement
    if metrics["avg_sharpe"] < MIN_SHARPE:
        raise RuntimeError("Sharpe below governance threshold.")

    if metrics["max_drawdown"] < MAX_DRAWDOWN:
        raise RuntimeError("Drawdown breach.")

    if metrics["num_windows"] < MIN_WINDOWS:
        raise RuntimeError("Insufficient validation windows.")

    compare_to_baseline(metrics)

    print("CI Walk-forward evaluation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())