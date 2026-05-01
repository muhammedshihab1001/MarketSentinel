"""
MarketSentinel — Outcome Service v1.0

Computes actual forward returns for predictions and updates the database
with outcome data. Enables agent accuracy tracking and evaluation.

Issue #26: Database schema extension for agent evaluation
- Fetches actual prices 5 days after prediction date
- Computes forward returns
- Marks direction correctness
- Updates predictions table with outcomes

Usage:
    from core.analytics.outcome_service import OutcomeService
    
    service = OutcomeService()
    results = service.compute_pending_outcomes()
    
    # Output:
    # {
    #   'processed': 247,
    #   'updated': 243,
    #   'errors': 4,
    #   'direction_accuracy': 0.67
    # }
"""

import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from core.data.market_data_service import MarketDataService
from core.db.repository import PredictionRepository, OHLCVRepository
from core.logging.logger import get_logger

logger = get_logger(__name__)

# Number of trading days to look forward for outcome
FORWARD_DAYS = 5

# Minimum price change to consider directional (avoid noise)
DIRECTION_THRESHOLD = 0.001  # 0.1%


class OutcomeService:
    """
    Computes actual outcomes for predictions and updates database.
    
    Workflow:
    1. Query predictions needing outcomes (date < today - 5 days, outcome_fetched_at IS NULL)
    2. For each prediction:
       - Fetch price at prediction date (close)
       - Fetch price at prediction date + 5 days (close)
       - Compute forward return: (price_t+5 - price_t) / price_t
       - Determine if direction was correct
       - Calculate prediction error
    3. Batch update predictions table
    """

    def __init__(self):
        self.market_service = MarketDataService()
        
        logger.info(
            "OutcomeService initialized | forward_days=%d",
            FORWARD_DAYS,
            extra={"component": "outcome_service", "function": "__init__"},
        )

    def compute_pending_outcomes(
        self,
        lookback_days: int = 30,
        batch_size: int = 100,
    ) -> Dict:
        """
        Compute outcomes for all predictions needing them.
        
        Args:
            lookback_days: How far back to search for predictions (default: 30)
            batch_size: Process predictions in batches (default: 100)
        
        Returns:
            Summary dict with processing statistics
        """
        start_time = pd.Timestamp.now()
        
        # Calculate date range
        today = pd.Timestamp.now(tz="UTC").normalize()
        cutoff_date = today - pd.Timedelta(days=FORWARD_DAYS)
        start_date = cutoff_date - pd.Timedelta(days=lookback_days)
        
        logger.info(
            "Starting outcome computation | cutoff=%s lookback=%d days",
            cutoff_date.strftime("%Y-%m-%d"),
            lookback_days,
            extra={"component": "outcome_service", "function": "compute_pending_outcomes"},
        )
        
        # Get predictions needing outcomes
        predictions = PredictionRepository.get_predictions_needing_outcomes(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=cutoff_date.strftime("%Y-%m-%d"),
        )
        
        if predictions is None or predictions.empty:
            logger.info(
                "No predictions needing outcomes",
                extra={"component": "outcome_service", "function": "compute_pending_outcomes"},
            )
            return {
                "processed": 0,
                "updated": 0,
                "errors": 0,
                "direction_accuracy": None,
                "mean_abs_error": None,
                "elapsed_seconds": 0,
            }
        
        total_predictions = len(predictions)
        logger.info(
            "Found %d predictions needing outcomes",
            total_predictions,
            extra={"component": "outcome_service", "function": "compute_pending_outcomes"},
        )
        
        # Process in batches
        all_outcomes = []
        error_count = 0
        
        for i in range(0, total_predictions, batch_size):
            batch = predictions.iloc[i:i + batch_size]
            
            logger.debug(
                "Processing batch %d-%d of %d",
                i,
                min(i + batch_size, total_predictions),
                total_predictions,
                extra={"component": "outcome_service", "function": "compute_pending_outcomes"},
            )
            
            batch_outcomes, batch_errors = self._compute_batch_outcomes(batch)
            all_outcomes.extend(batch_outcomes)
            error_count += batch_errors
        
        # Update database
        if all_outcomes:
            updated = PredictionRepository.update_prediction_outcomes(all_outcomes)
        else:
            updated = 0
        
        # Calculate statistics
        stats = self._calculate_statistics(all_outcomes)
        
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        
        result = {
            "processed": total_predictions,
            "updated": updated,
            "errors": error_count,
            "direction_accuracy": stats.get("direction_accuracy"),
            "mean_abs_error": stats.get("mean_abs_error"),
            "elapsed_seconds": round(elapsed, 2),
        }
        
        logger.info(
            "Outcome computation complete | processed=%d updated=%d errors=%d "
            "accuracy=%.2f%% mae=%.4f elapsed=%.1fs",
            result["processed"],
            result["updated"],
            result["errors"],
            (result["direction_accuracy"] or 0) * 100,
            result["mean_abs_error"] or 0,
            result["elapsed_seconds"],
            extra={"component": "outcome_service", "function": "compute_pending_outcomes"},
        )
        
        return result

    def _compute_batch_outcomes(
        self,
        predictions: pd.DataFrame,
    ) -> Tuple[List[Dict], int]:
        """
        Compute outcomes for a batch of predictions.
        
        Args:
            predictions: DataFrame with prediction records
        
        Returns:
            Tuple of (outcome_records, error_count)
        """
        outcomes = []
        errors = 0
        
        # Group by ticker for efficient price fetching
        for ticker, group in predictions.groupby("ticker"):
            
            try:
                ticker_outcomes = self._compute_ticker_outcomes(ticker, group)
                outcomes.extend(ticker_outcomes)
                
            except Exception as e:
                logger.warning(
                    "Failed to compute outcomes for ticker=%s | error=%s",
                    ticker,
                    str(e),
                    extra={"component": "outcome_service", "function": "_compute_batch_outcomes"},
                )
                errors += len(group)
        
        return outcomes, errors

    def _compute_ticker_outcomes(
        self,
        ticker: str,
        predictions: pd.DataFrame,
    ) -> List[Dict]:
        """
        Compute outcomes for all predictions of a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            predictions: DataFrame with predictions for this ticker
        
        Returns:
            List of outcome records
        """
        outcomes = []
        
        # Get date range for price fetching
        pred_dates = pd.to_datetime(predictions["date"])
        min_date = pred_dates.min()
        max_date = pred_dates.max() + pd.Timedelta(days=FORWARD_DAYS + 2)  # Extra buffer
        
        # Fetch price data
        try:
            prices = self.market_service.get_price_data(
                ticker=ticker,
                start_date=min_date.strftime("%Y-%m-%d"),
                end_date=max_date.strftime("%Y-%m-%d"),
                min_history=5,
            )
        except Exception as e:
            logger.warning(
                "Price fetch failed for ticker=%s | error=%s",
                ticker,
                str(e),
                extra={"component": "outcome_service", "function": "_compute_ticker_outcomes"},
            )
            return []
        
        if prices is None or prices.empty:
            return []
        
        # Create price lookup dict (date -> close price)
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        price_lookup = dict(zip(prices["date"], prices["close"]))
        
        # Compute outcomes for each prediction
        for _, pred in predictions.iterrows():
            
            outcome = self._compute_single_outcome(
                prediction=pred,
                price_lookup=price_lookup,
            )
            
            if outcome:
                outcomes.append(outcome)
        
        return outcomes

    def _compute_single_outcome(
        self,
        prediction: pd.Series,
        price_lookup: Dict[pd.Timestamp, float],
    ) -> Optional[Dict]:
        """
        Compute outcome for a single prediction.
        
        Args:
            prediction: Prediction record (Series)
            price_lookup: Dict mapping date -> close price
        
        Returns:
            Outcome dict or None if cannot compute
        """
        pred_id = prediction.get("id")
        ticker = prediction.get("ticker")
        pred_date = pd.to_datetime(prediction["date"]).normalize()
        
        # Get prediction signal/score
        predicted_signal = prediction.get("signal")  # LONG / SHORT / NEUTRAL
        predicted_score = float(prediction.get("raw_model_score", 0.0))
        
        # Get prices at t and t+5
        price_t = price_lookup.get(pred_date)
        
        if price_t is None:
            logger.debug(
                "No price found for prediction date | ticker=%s date=%s",
                ticker,
                pred_date.strftime("%Y-%m-%d"),
                extra={"component": "outcome_service", "function": "_compute_single_outcome"},
            )
            return None
        
        # Find actual price at t+5 (may not be exactly 5 days due to weekends/holidays)
        target_date = pred_date + pd.Timedelta(days=FORWARD_DAYS)
        price_t5 = None
        
        # Search forward up to 10 calendar days to find next trading day
        for offset in range(FORWARD_DAYS, FORWARD_DAYS + 6):
            check_date = pred_date + pd.Timedelta(days=offset)
            price_t5 = price_lookup.get(check_date)
            if price_t5 is not None:
                break
        
        if price_t5 is None:
            logger.debug(
                "No price found for target date | ticker=%s target=%s",
                ticker,
                target_date.strftime("%Y-%m-%d"),
                extra={"component": "outcome_service", "function": "_compute_single_outcome"},
            )
            return None
        
        # Compute forward return
        actual_return = (price_t5 - price_t) / price_t
        
        # Determine actual direction
        if actual_return > DIRECTION_THRESHOLD:
            actual_direction = "LONG"
        elif actual_return < -DIRECTION_THRESHOLD:
            actual_direction = "SHORT"
        else:
            actual_direction = "NEUTRAL"
        
        # Check if direction was correct
        direction_correct = (predicted_signal == actual_direction)
        
        # If prediction was NEUTRAL, consider it correct if actual was also NEUTRAL
        # or if the magnitude was small
        if predicted_signal == "NEUTRAL":
            direction_correct = abs(actual_return) <= DIRECTION_THRESHOLD * 2
        
        # Compute prediction error (absolute difference)
        prediction_error = abs(predicted_score - actual_return)
        
        # Build outcome record
        outcome = {
            "id": pred_id,
            "ticker": ticker,
            "date": pred_date.strftime("%Y-%m-%d"),
            "actual_forward_return": round(actual_return, 6),
            "direction_correct": direction_correct,
            "prediction_error": round(prediction_error, 6),
            "outcome_fetched_at": pd.Timestamp.now(tz="UTC"),
        }
        
        logger.debug(
            "Computed outcome | ticker=%s date=%s predicted=%s actual=%s correct=%s return=%.4f",
            ticker,
            pred_date.strftime("%Y-%m-%d"),
            predicted_signal,
            actual_direction,
            direction_correct,
            actual_return,
            extra={"component": "outcome_service", "function": "_compute_single_outcome"},
        )
        
        return outcome

    def _calculate_statistics(self, outcomes: List[Dict]) -> Dict:
        """
        Calculate summary statistics from outcomes.
        
        Args:
            outcomes: List of outcome records
        
        Returns:
            Dict with statistics
        """
        if not outcomes:
            return {
                "direction_accuracy": None,
                "mean_abs_error": None,
            }
        
        # Direction accuracy
        correct_count = sum(1 for o in outcomes if o.get("direction_correct"))
        direction_accuracy = correct_count / len(outcomes)
        
        # Mean absolute error
        errors = [o.get("prediction_error", 0) for o in outcomes]
        mean_abs_error = float(np.mean(errors))
        
        return {
            "direction_accuracy": round(direction_accuracy, 4),
            "mean_abs_error": round(mean_abs_error, 6),
        }

    def compute_single_prediction_outcome(
        self,
        prediction_id: int,
    ) -> Optional[Dict]:
        """
        Compute outcome for a single prediction by ID.
        
        Useful for testing or manual correction.
        
        Args:
            prediction_id: Database ID of prediction
        
        Returns:
            Outcome dict or None if failed
        """
        # Get prediction from database
        # This would require adding a get_prediction_by_id method to repository
        # For now, this is a placeholder for future enhancement
        
        logger.info(
            "Single prediction outcome | id=%d",
            prediction_id,
            extra={"component": "outcome_service", "function": "compute_single_prediction_outcome"},
        )
        
        # Implementation would go here
        return None