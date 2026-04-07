"""
MarketSentinel Institutional Evaluation Runner v4.2
Aligned with Walk-Forward Portfolio Validation
Hybrid-Ready Governance Layer
"""

import sys
import json
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
from core.db.engine import init_db
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.run_evaluation")


# =========================================================
# GOVERNANCE THRESHOLDS
# =========================================================

MIN_SHARPE = 0.10
MAX_DRAWDOWN = -0.60
MIN_WINDOWS = 5
MAX_SHARPE_DEGRADATION = 0.10

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
        logger.warning(
            "Price fetch failed | tickers=%s | count=%d | function=build_dataset",
            fetch_failures,
            len(fetch_failures),
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

    logger.info(
        "Evaluation dataset built | rows=%d | tickers=%d | dates=%d | function=build_dataset",
        len(df),
        df["ticker"].nunique(),
        df["date"].nunique(),
    )

    return df


# =========================================================
# BASELINE COMPARISON
# =========================================================

def compare_to_baseline(metrics):

    baseline_path = Path("artifacts/xgboost/baseline_contract.json")

    if not baseline_path.exists():
        logger.info("No baseline contract found — skipping comparison.")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_metrics = baseline.get("metrics", {})
    baseline_sharpe = baseline_metrics.get("avg_sharpe")

    if baseline_sharpe is None:
        return

    current_sharpe = metrics.get("avg_sharpe", 0)

    if current_sharpe < baseline_sharpe - MAX_SHARPE_DEGRADATION:

        raise RuntimeError(
            f"Sharpe degraded vs baseline. "
            f"Current={current_sharpe:.4f} "
            f"Baseline={baseline_sharpe:.4f}"
        )

    logger.info(
        "Baseline comparison passed | current_sharpe=%.4f | baseline_sharpe=%.4f | function=compare_to_baseline",
        current_sharpe,
        baseline_sharpe,
    )


# =========================================================
# MAIN
# =========================================================

def main() -> int:

    logger.info("Starting Institutional Walk-Forward CI Evaluation | function=main")

    init_env()

    # ── Init DB before any data access ──────────────────
    logger.info("Initialising database connection | function=main")

    try:
        init_db()
        logger.info("Database ready | function=main")
    except Exception as e:
        logger.warning(
            "DB init failed — evaluation will use cached data if available | error=%s | function=main",
            e,
        )

    loader = ModelLoader()

    logger.info(
        "Evaluating model | version=%s | schema_signature=%.12s | function=main",
        loader.xgb_version,
        loader.schema_signature,
    )

    df = build_dataset()

    X = validate_feature_schema(
        df.loc[:, MODEL_FEATURES],
        mode="strict_contract"
    )

    # =====================================================
    # DRIFT CHECK
    # =====================================================

    drift_detector = DriftDetector()

    drift_score = drift_detector.compute_drift(X)

    logger.info(
        "Drift severity score | score=%s | function=main",
        drift_score,
    )

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

    logger.info("Walk-forward metrics | metrics=%s | function=main", metrics)

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

    logger.info("CI Walk-forward evaluation PASSED | function=main")

    return 0


if __name__ == "__main__":
    sys.exit(main())
