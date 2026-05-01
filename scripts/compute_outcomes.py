
"""
MarketSentinel — Compute Prediction Outcomes (CLI)

Fetches actual forward returns for predictions and updates the database.
Can be run manually or scheduled as a cron job.

Usage:
    # Compute outcomes for predictions from last 30 days
    python scripts/compute_outcomes.py
    
    # Specify custom lookback period
    python scripts/compute_outcomes.py --lookback-days 60
    
    # Dry run (show what would be processed)
    python scripts/compute_outcomes.py --dry-run
    
    # Process only specific date range
    python scripts/compute_outcomes.py --start-date 2026-01-01 --end-date 2026-01-31

Cron Example (run daily at 6 AM):
    0 6 * * * cd /path/to/marketsentinel && python scripts/compute_outcomes.py >> logs/outcomes.log 2>&1

Docker Example:
    docker-compose exec api python scripts/compute_outcomes.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from core.analytics.outcome_service import OutcomeService
from core.logging.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Compute actual outcomes for predictions"
    )
    
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="How many days back to search for predictions (default: 30)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Process predictions in batches of this size (default: 100)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without updating database",
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Custom start date (YYYY-MM-DD) - overrides lookback-days",
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Custom end date (YYYY-MM-DD) - overrides automatic cutoff",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Initialize service
    service = OutcomeService()
    
    print("=" * 60)
    print("MarketSentinel — Outcome Computation")
    print("=" * 60)
    print()
    
    if args.dry_run:
        print(" DRY RUN MODE - No database changes will be made")
        print()
    
    # Show configuration
    print(f"Configuration:")
    print(f"  Lookback Days: {args.lookback_days}")
    print(f"  Batch Size: {args.batch_size}")
    
    if args.start_date:
        print(f"  Start Date: {args.start_date}")
    if args.end_date:
        print(f"  End Date: {args.end_date}")
    
    print()
    print("-" * 60)
    print()
    
    # Compute outcomes
    try:
        if args.dry_run:
            # In dry run, just query what would be processed
            from core.db.repository import PredictionRepository
            
            today = pd.Timestamp.now(tz="UTC").normalize()
            cutoff_date = today - pd.Timedelta(days=5)
            
            if args.start_date:
                start = pd.Timestamp(args.start_date)
            else:
                start = cutoff_date - pd.Timedelta(days=args.lookback_days)
            
            if args.end_date:
                end = pd.Timestamp(args.end_date)
            else:
                end = cutoff_date
            
            predictions = PredictionRepository.get_predictions_needing_outcomes(
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
            
            if predictions is None or predictions.empty:
                print(" No predictions need outcomes")
                return 0
            
            print(f" Would process {len(predictions)} predictions:")
            print()
            print(f"  Date Range: {predictions['date'].min()} → {predictions['date'].max()}")
            print(f"  Unique Tickers: {predictions['ticker'].nunique()}")
            print(f"  Total Records: {len(predictions)}")
            print()
            print("Top 10 tickers by prediction count:")
            print(predictions['ticker'].value_counts().head(10))
            print()
            print(" Run without --dry-run to process these predictions")
            
            return 0
        
        else:
            # Actually compute outcomes
            results = service.compute_pending_outcomes(
                lookback_days=args.lookback_days,
                batch_size=args.batch_size,
            )
            
            # Display results
            print(" Outcome Computation Complete")
            print()
            print("Results:")
            print(f"  Processed: {results['processed']} predictions")
            print(f"  Updated: {results['updated']} records")
            print(f"  Errors: {results['errors']} failures")
            
            if results['direction_accuracy'] is not None:
                print(f"  Direction Accuracy: {results['direction_accuracy']:.2%}")
            
            if results['mean_abs_error'] is not None:
                print(f"  Mean Absolute Error: {results['mean_abs_error']:.6f}")
            
            print(f"  Elapsed Time: {results['elapsed_seconds']:.2f}s")
            print()
            
            if results['updated'] > 0:
                print("Successfully updated {results['updated']} predictions with outcomes")
            else:
                print("No predictions were updated (all may already have outcomes)")
            
            return 0
    
    except KeyboardInterrupt:
        print()
        print("Interrupted by user")
        return 1
    
    except Exception as e:
        print()
        print(f" Error: {e}")
        logger.exception("Outcome computation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())