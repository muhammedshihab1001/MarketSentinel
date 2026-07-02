#!/usr/bin/env python3
"""
MarketSentinel — Update Agent Weights

Updates agent weights in config based on recent performance.
Runs weekly after agent evaluation.

Usage:
    python scripts/update_agent_weights.py
    python scripts/update_agent_weights.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from core.db.repository import PredictionRepository
from core.logging.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = project_root / "config" / "agent_weights.json"


def main():
    parser = argparse.ArgumentParser(description="Update agent weights from performance")
    parser.add_argument("--dry-run", action="store_true", help="Show new weights without saving")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MarketSentinel — Agent Weight Update")
    print("=" * 60)
    print()
    
    # Load current config
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    current_weights = config["weights"]
    print("Current Weights:")
    for agent, weight in current_weights.items():
        print(f"  {agent}: {weight:.2f}")
    print()
    
    # Get latest performance
    try:
        perf = PredictionRepository.get_agent_performance(days=7)
        
        if perf is None or perf.empty:
            print("  No recent performance data - keeping current weights")
            return 0
        
        # Get most recent evaluation per agent
        latest = perf.sort_values("evaluation_date").groupby("agent_name").tail(1)
        
        print("Recent Performance (last 7 days):")
        for _, row in latest.iterrows():
            print(f"  {row['agent_name']}:")
            print(f"    Accuracy: {row['direction_accuracy']:.2%}")
            print(f"    Sharpe: {row['sharpe_ratio']:.2f}")
        print()
        
        # Simple performance-based adjustment
        new_weights = {}
        for agent in ["signal_agent", "technical_agent", "raw_model"]:
            agent_perf = latest[latest["agent_name"] == agent]
            
            if agent_perf.empty:
                new_weights[agent] = current_weights.get(agent, 0.30)
                continue
            
            accuracy = agent_perf.iloc[0]["direction_accuracy"]
            sharpe = agent_perf.iloc[0]["sharpe_ratio"]
            
            # Score = 70% accuracy + 30% Sharpe (normalized)
            score = 0.7 * accuracy + 0.3 * min(sharpe / 2.0, 1.0)
            new_weights[agent] = score
        
        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}
        
        # Apply bounds
        min_w = config["bounds"]["min_weight"]
        max_w = config["bounds"]["max_weight"]
        new_weights = {k: max(min_w, min(max_w, v)) for k, v in new_weights.items()}
        
        # Re-normalize after bounds
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}
        
        print("New Weights:")
        for agent, weight in new_weights.items():
            old = current_weights[agent]
            change = weight - old
            print(f"  {agent}: {weight:.2f} ({change:+.2f})")
        print()
        
        if args.dry_run:
            print(" DRY RUN - Weights not saved")
            return 0
        
        # Update config
        config["weights"] = new_weights
        config["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
        
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        
        print(" Weights updated and saved to config/agent_weights.json")
        return 0
        
    except Exception as e:
        print(f" Error: {e}")
        logger.exception("Weight update failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())