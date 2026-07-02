"""
MarketSentinel -- Agent Evaluator v1.1

Computes agent accuracy and performance metrics by comparing predictions
to actual outcomes. Enables performance-based agent weight adjustment.

Issue #27: Agent accuracy tracking and evaluation
- Computes direction accuracy per agent
- Confidence calibration analysis
- Sharpe contribution by agent
- Stores results in agent_performance table

FIX v1.1:
  _evaluate_signal_agent() was using direction_correct from DB.
  direction_correct in DB = hybrid model signal vs actual direction.
  Signal agent accuracy must be computed from signal_agent_signal
  vs actual direction derived from actual_forward_return.

  _evaluate_raw_model() had same bug. Raw model direction must be
  derived from sign(raw_model_score), not from DB direction_correct.

Usage:
    from core.analytics.agent_evaluator import AgentEvaluator

    evaluator = AgentEvaluator()
    results = evaluator.evaluate_agents(days=30)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.db.repository import PredictionRepository
from core.logging.logger import get_logger

logger = get_logger(__name__)

# Minimum predictions needed for meaningful statistics
MIN_PREDICTIONS = 20

# Threshold to determine direction from return (matches outcome_service.py)
DIRECTION_THRESHOLD = 0.001


def _actual_direction_series(returns: pd.Series) -> pd.Series:
    """
    Convert actual forward returns to direction labels.

    Matches the same threshold used in OutcomeService so comparisons
    are consistent.

    Args:
        returns: Series of actual forward returns

    Returns:
        Series of "LONG" / "SHORT" / "NEUTRAL" strings
    """
    directions = pd.Series("NEUTRAL", index=returns.index)
    directions[returns > DIRECTION_THRESHOLD] = "LONG"
    directions[returns < -DIRECTION_THRESHOLD] = "SHORT"
    return directions


class AgentEvaluator:
    """
    Evaluates agent performance by analyzing prediction accuracy.

    Each agent is evaluated independently against actual outcomes:
      - signal_agent: signal_agent_signal vs actual direction
      - technical_agent: technical_agent_bias vs actual direction
      - model_only: sign(raw_model_score) vs actual direction

    NOTE: direction_correct in the DB represents the HYBRID model
    signal vs actual direction -- it is NOT used for per-agent
    accuracy. Each agent computes its own directional correctness.
    """

    def __init__(self):
        logger.info(
            "AgentEvaluator initialized",
            extra={"component": "agent_evaluator", "function": "__init__"},
        )

    def evaluate_agents(
        self,
        days: int = 30,
        min_predictions: int = MIN_PREDICTIONS,
    ) -> Dict:
        """
        Evaluate all agents over lookback period.

        Args:
            days: Lookback period in days (default: 30)
            min_predictions: Minimum predictions needed (default: 20)

        Returns:
            Dict with agent metrics
        """
        start_time = pd.Timestamp.now()

        today = pd.Timestamp.now(tz="UTC").normalize()
        start_date = today - pd.Timedelta(days=days)

        logger.info(
            "Starting agent evaluation | start_date=%s days=%d",
            start_date.strftime("%Y-%m-%d"),
            days,
            extra={"component": "agent_evaluator", "function": "evaluate_agents"},
        )

        predictions = PredictionRepository.get_predictions_with_outcomes(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
        )

        if predictions is None or predictions.empty:
            logger.warning(
                "No predictions with outcomes found for evaluation",
                extra={"component": "agent_evaluator", "function": "evaluate_agents"},
            )
            return {
                "evaluated": False,
                "reason": "No predictions with outcomes",
                "min_predictions": min_predictions,
            }

        total_predictions = len(predictions)
        logger.info(
            "Found %d predictions with outcomes for evaluation",
            total_predictions,
            extra={"component": "agent_evaluator", "function": "evaluate_agents"},
        )

        if total_predictions < min_predictions:
            logger.warning(
                "Insufficient predictions for evaluation | have=%d need=%d",
                total_predictions,
                min_predictions,
                extra={"component": "agent_evaluator", "function": "evaluate_agents"},
            )
            return {
                "evaluated": False,
                "reason": f"Insufficient data ({total_predictions} < {min_predictions})",
                "min_predictions": min_predictions,
            }

        results = {}

        signal_metrics = self._evaluate_signal_agent(predictions)
        if signal_metrics:
            results["signal_agent"] = signal_metrics

        technical_metrics = self._evaluate_technical_agent(predictions)
        if technical_metrics:
            results["technical_agent"] = technical_metrics

        model_metrics = self._evaluate_raw_model(predictions)
        if model_metrics:
            results["model_only"] = model_metrics

        if results:
            stored = self._store_performance(results, start_date, today)
            results["stored"] = stored

        elapsed = (pd.Timestamp.now() - start_time).total_seconds()

        logger.info(
            "Agent evaluation complete | predictions=%d agents=%d elapsed=%.1fs",
            total_predictions,
            len([k for k in results if k not in ("stored",)]),
            elapsed,
            extra={"component": "agent_evaluator", "function": "evaluate_agents"},
        )

        results["evaluated"] = True
        results["total_predictions"] = total_predictions
        results["period_days"] = days
        results["elapsed_seconds"] = round(elapsed, 2)

        return results

    def _evaluate_signal_agent(self, predictions: pd.DataFrame) -> Optional[Dict]:
        """
        Evaluate SignalAgent performance.

        FIX v1.1: Computes direction correctness by comparing
        signal_agent_signal directly vs actual direction from
        actual_forward_return. Does NOT use direction_correct from
        DB (which is hybrid model accuracy, not signal agent accuracy).

        Args:
            predictions: DataFrame with predictions and outcomes

        Returns:
            Dict with signal agent metrics or None
        """
        agent_preds = predictions[
            predictions["signal_agent_signal"].notna()
            & predictions["actual_forward_return"].notna()
        ].copy()

        if agent_preds.empty:
            logger.debug("No signal agent predictions found")
            return None

        # Compute actual direction from actual_forward_return
        actual_dir = _actual_direction_series(agent_preds["actual_forward_return"])

        # Compare signal_agent_signal vs actual direction
        # Handle NEUTRAL: correct if actual return was also within threshold
        agent_signal = agent_preds["signal_agent_signal"]
        agent_correct = agent_signal == actual_dir

        # NEUTRAL predictions: correct if |return| <= threshold * 2
        neutral_mask = agent_signal == "NEUTRAL"
        agent_correct[neutral_mask] = (
            agent_preds.loc[neutral_mask, "actual_forward_return"].abs()
            <= DIRECTION_THRESHOLD * 2
        )

        direction_accuracy = agent_correct.sum() / len(agent_preds)

        confidence_calibration = self._compute_confidence_calibration(
            agent_preds["signal_agent_confidence"],
            agent_correct,
        )

        sharpe = self._compute_sharpe(
            agent_preds["actual_forward_return"],
            agent_preds["signal_agent_score"],
        )

        avg_score = float(agent_preds["signal_agent_score"].mean())

        metrics = {
            "direction_accuracy": round(float(direction_accuracy), 4),
            "confidence_calibration": round(confidence_calibration, 4),
            "sharpe": round(sharpe, 4),
            "avg_score": round(avg_score, 4),
            "num_predictions": len(agent_preds),
        }

        logger.debug(
            "Signal agent metrics | accuracy=%.2f%% sharpe=%.2f n=%d",
            direction_accuracy * 100,
            sharpe,
            len(agent_preds),
            extra={"component": "agent_evaluator", "function": "_evaluate_signal_agent"},
        )

        return metrics

    def _evaluate_technical_agent(self, predictions: pd.DataFrame) -> Optional[Dict]:
        """
        Evaluate TechnicalRiskAgent performance.

        technical_agent_bias (bullish/bearish/neutral) is compared
        directly to actual direction. This was already correct in v1.0.

        Args:
            predictions: DataFrame with predictions and outcomes

        Returns:
            Dict with technical agent metrics or None
        """
        agent_preds = predictions[
            predictions["technical_agent_score"].notna()
            & predictions["actual_forward_return"].notna()
        ].copy()

        if agent_preds.empty:
            logger.debug("No technical agent predictions found")
            return None

        # Map bias to direction
        agent_preds["tech_direction"] = agent_preds["technical_agent_bias"].map({
            "bullish": "LONG",
            "bearish": "SHORT",
            "neutral": "NEUTRAL",
        })

        # Actual direction from return
        actual_dir = _actual_direction_series(agent_preds["actual_forward_return"])
        direction_correct = (agent_preds["tech_direction"] == actual_dir).sum()
        direction_accuracy = direction_correct / len(agent_preds)

        sharpe = self._compute_sharpe(
            agent_preds["actual_forward_return"],
            agent_preds["technical_agent_score"],
        )

        avg_score = float(agent_preds["technical_agent_score"].mean())

        metrics = {
            "direction_accuracy": round(float(direction_accuracy), 4),
            "sharpe": round(sharpe, 4),
            "avg_score": round(avg_score, 4),
            "num_predictions": len(agent_preds),
        }

        logger.debug(
            "Technical agent metrics | accuracy=%.2f%% sharpe=%.2f n=%d",
            direction_accuracy * 100,
            sharpe,
            len(agent_preds),
            extra={"component": "agent_evaluator", "function": "_evaluate_technical_agent"},
        )

        return metrics

    def _evaluate_raw_model(self, predictions: pd.DataFrame) -> Optional[Dict]:
        """
        Evaluate raw XGBoost model performance (baseline).

        FIX v1.1: Derives model direction from sign(raw_model_score)
        instead of using direction_correct from DB. direction_correct
        in DB is computed from the hybrid signal, not the raw score.

        Args:
            predictions: DataFrame with predictions and outcomes

        Returns:
            Dict with model metrics or None
        """
        model_preds = predictions[
            predictions["raw_model_score"].notna()
            & predictions["actual_forward_return"].notna()
        ].copy()

        if model_preds.empty:
            logger.debug("No model predictions found")
            return None

        # Derive model direction from sign of raw_model_score
        model_direction = pd.Series("NEUTRAL", index=model_preds.index)
        model_direction[model_preds["raw_model_score"] > 0.01] = "LONG"
        model_direction[model_preds["raw_model_score"] < -0.01] = "SHORT"

        # Actual direction from return
        actual_dir = _actual_direction_series(model_preds["actual_forward_return"])
        direction_correct = (model_direction == actual_dir).sum()
        direction_accuracy = direction_correct / len(model_preds)

        sharpe = self._compute_sharpe(
            model_preds["actual_forward_return"],
            model_preds["raw_model_score"],
        )

        mae = float(model_preds["prediction_error"].mean())

        metrics = {
            "direction_accuracy": round(float(direction_accuracy), 4),
            "sharpe": round(sharpe, 4),
            "mean_absolute_error": round(mae, 6),
            "num_predictions": len(model_preds),
        }

        logger.debug(
            "Raw model metrics | accuracy=%.2f%% sharpe=%.2f n=%d",
            direction_accuracy * 100,
            sharpe,
            len(model_preds),
            extra={"component": "agent_evaluator", "function": "_evaluate_raw_model"},
        )

        return metrics

    def _compute_confidence_calibration(
        self,
        predicted_confidence: pd.Series,
        actual_correct: pd.Series,
    ) -> float:
        """
        Compute confidence calibration score.

        Args:
            predicted_confidence: Predicted confidence levels (0-1)
            actual_correct: Boolean series of correct predictions

        Returns:
            Calibration score (0-1, higher is better)
        """
        if predicted_confidence.isna().all() or actual_correct.isna().all():
            return 0.0

        bins = [0, 0.3, 0.5, 0.7, 1.0]
        labels = ["low", "medium", "high", "very_high"]

        df = pd.DataFrame({
            "confidence": predicted_confidence,
            "correct": actual_correct,
        }).dropna()

        if df.empty:
            return 0.0

        df["conf_bin"] = pd.cut(df["confidence"], bins=bins, labels=labels)

        calibration_error = 0.0
        for bin_label in labels:
            bin_data = df[df["conf_bin"] == bin_label]
            if len(bin_data) < 5:
                continue
            avg_confidence = bin_data["confidence"].mean()
            actual_accuracy = bin_data["correct"].mean()
            calibration_error += abs(avg_confidence - actual_accuracy)

        calibration_score = max(0.0, 1.0 - (calibration_error / len(labels)))
        return calibration_score

    def _compute_sharpe(
        self,
        actual_returns: pd.Series,
        predicted_scores: pd.Series,
    ) -> float:
        """
        Compute annualized Sharpe ratio from predicted scores.

        Strategy returns = actual_return * sign(predicted_score).

        Args:
            actual_returns: Actual forward returns
            predicted_scores: Predicted scores (-1 to 1)

        Returns:
            Annualized Sharpe ratio (252 trading days)
        """
        df = pd.DataFrame({
            "returns": actual_returns,
            "scores": predicted_scores,
        }).dropna()

        if df.empty or len(df) < 10:
            return 0.0

        df["strategy_returns"] = df["returns"] * np.sign(df["scores"])

        mean_return = df["strategy_returns"].mean()
        std_return = df["strategy_returns"].std()

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        return float((mean_return / std_return) * np.sqrt(252))

    def _store_performance(
        self,
        results: Dict,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> bool:
        """
        Store agent performance results in database.

        Args:
            results: Dict with agent metrics
            start_date: Period start
            end_date: Period end

        Returns:
            True if stored successfully
        """
        _meta_keys = {
            "stored", "evaluated", "total_predictions",
            "period_days", "elapsed_seconds",
        }

        try:
            records = []

            for agent_name, metrics in results.items():
                if agent_name in _meta_keys:
                    continue

                record = {
                    "agent_name": agent_name,
                    "evaluation_date": end_date.strftime("%Y-%m-%d"),
                    "period_start": start_date.strftime("%Y-%m-%d"),
                    "period_end": end_date.strftime("%Y-%m-%d"),
                    "direction_accuracy": metrics.get("direction_accuracy"),
                    "sharpe_ratio": metrics.get("sharpe"),
                    "num_predictions": metrics.get("num_predictions"),
                    "avg_score": metrics.get("avg_score"),
                    "confidence_calibration": metrics.get("confidence_calibration"),
                    "mean_absolute_error": metrics.get("mean_absolute_error"),
                }

                records.append(record)

            if records:
                stored = PredictionRepository.store_agent_performance(records)
                logger.info(
                    "Agent performance stored | records=%d",
                    stored,
                    extra={
                        "component": "agent_evaluator",
                        "function": "_store_performance",
                    },
                )
                return stored > 0

            return False

        except Exception as e:
            logger.warning(
                "Failed to store agent performance | error=%s",
                str(e),
                extra={
                    "component": "agent_evaluator",
                    "function": "_store_performance",
                },
            )
            return False