"""
MarketSentinel Institutional Evaluation Runner v4.1
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

# DriftDetector severity scale = 0–15
MAX_ALLOWED_DRIFT = 8


# =========================================================
# DATASET BUILD
# =========================================================

def build_dataset():

    market_data = MarketDataService()
    universe = MarketUniverse.get_universe()

    start_date, end_date = MarketTime.window_for("xgboost")

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

        except Exception:
            fetch_failures.append(ticker)

    if not price_frames:
        raise RuntimeError("No evaluation price data fetched.")

    if fetch_failures:
        print(
            f"WARNING: price fetch failed for "
            f"{len(fetch_failures)} tickers: {fetch_failures}"
        )

    combined_prices = pd.concat(price_frames, ignore_index=True)

    df = FeatureEngineer.build_feature_pipeline(
        combined_prices,
        training=False
    )

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if len(df) < 2000:
        raise RuntimeError(
            f"Dataset too small for evaluation: {len(df)} rows, need 2000."
        )

    print(
        f"Evaluation dataset built | rows={len(df)} "
        f"tickers={df['ticker'].nunique()} "
        f"dates={df['date'].nunique()}"
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

    if metrics.get("avg_sharpe", 0) < baseline_sharpe - MAX_SHARPE_DEGRADATION:

        raise RuntimeError(
            f"Sharpe degraded vs baseline. "
            f"Current={metrics.get('avg_sharpe'):.4f} "
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

    df = build_dataset()

    # Strict schema validation
    X = validate_feature_schema(
        df.loc[:, MODEL_FEATURES],
        mode="strict_contract"
    )

    # =====================================================
    # DRIFT CHECK
    # =====================================================

    drift_detector = DriftDetector()

    drift_score = drift_detector.compute_drift(X)

    print(f"Drift severity score: {drift_score}")

    if drift_score > MAX_ALLOWED_DRIFT:

        raise RuntimeError(
            f"Drift exceeded governance limit "
            f"(score={drift_score}, max={MAX_ALLOWED_DRIFT})"
        )

    # =====================================================
    # WALK FORWARD VALIDATION
    # =====================================================

    validator = WalkForwardValidator(
        model_trainer=retrain_model
    )

    metrics = validator.run(df.copy())

    print("Walk-forward metrics:", metrics)

    # =====================================================
    # GOVERNANCE RULES
    # =====================================================

    if metrics.get("avg_sharpe", 0) < MIN_SHARPE:
        raise RuntimeError("Sharpe below governance threshold.")

    if metrics.get("max_drawdown", 0) < MAX_DRAWDOWN:
        raise RuntimeError("Drawdown breach.")

    if metrics.get("num_windows", 0) < MIN_WINDOWS:
        raise RuntimeError("Insufficient validation windows.")

    compare_to_baseline(metrics)

    print("CI Walk-forward evaluation PASSED.")

    return 0


if __name__ == "__main__":
    sys.exit(main())