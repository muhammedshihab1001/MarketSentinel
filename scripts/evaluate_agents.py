#!/usr/bin/env python3
"""
MarketSentinel — Evaluate Agent Performance (CLI)

Computes agent accuracy and performance metrics by comparing predictions
to actual outcomes. Updates Prometheus gauges after evaluation.

Usage:
    python scripts/evaluate_agents.py
    python scripts/evaluate_agents.py --days 60
    python scripts/evaluate_agents.py --dry-run

Issue #27: Agent evaluation
Issue #28: Updates Prometheus gauges after evaluation
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.analytics.agent_evaluator import AgentEvaluator
from core.logging.logger import get_logger

logger = get_logger(__name__)


def _update_prometheus_gauges(results: dict) -> None:
    """
    Update Prometheus gauges with agent evaluation results.
    Issue #28: Push metrics after every evaluation run.
    """
    try:
        from app.monitoring.metrics import (
            AGENT_DIRECTION_ACCURACY,
            AGENT_SHARPE_CONTRIBUTION,
            AGENT_NUM_PREDICTIONS,
            AGENT_CONFIDENCE_CALIBRATION,
            MODEL_ONLY_SHARPE,
        )

        agent_keys = ["signal_agent", "technical_agent", "model_only"]

        for agent in agent_keys:
            if agent not in results:
                continue

            metrics = results[agent]

            # Direction accuracy gauge
            acc = metrics.get("direction_accuracy", 0.0)
            AGENT_DIRECTION_ACCURACY.labels(agent_name=agent).set(acc)

            # Sharpe gauge
            sharpe = metrics.get("sharpe", 0.0)
            AGENT_SHARPE_CONTRIBUTION.labels(agent_name=agent).set(sharpe)

            # Num predictions gauge
            n = metrics.get("num_predictions", 0)
            AGENT_NUM_PREDICTIONS.labels(agent_name=agent).set(n)

            # Confidence calibration (signal_agent only)
            if "confidence_calibration" in metrics:
                cal = metrics.get("confidence_calibration", 0.0)
                AGENT_CONFIDENCE_CALIBRATION.labels(agent_name=agent).set(cal)

        # Model-only Sharpe baseline
        if "model_only" in results:
            MODEL_ONLY_SHARPE.set(results["model_only"].get("sharpe", 0.0))

        logger.info(
            "Prometheus gauges updated | agents=%d",
            len([k for k in agent_keys if k in results]),
        )

    except ImportError:
        # Prometheus not available in this environment (e.g. standalone scheduler)
        logger.debug("Prometheus metrics not available — skipping gauge update")
    except Exception as e:
        # Non-blocking — evaluation still succeeds even if gauge update fails
        logger.warning("Gauge update failed (non-blocking): %s", e)


def _update_weight_gauges(weights: dict) -> None:
    """
    Update AGENT_WEIGHT gauges from agent_weights.json.
    Issue #28: Reflect current weights in Prometheus.
    """
    try:
        from app.monitoring.metrics import AGENT_WEIGHT

        for agent, weight in weights.items():
            AGENT_WEIGHT.labels(agent_name=agent).set(weight)

        logger.info("Agent weight gauges updated | weights=%s", weights)

    except ImportError:
        logger.debug("Prometheus metrics not available — skipping weight gauge update")
    except Exception as e:
        logger.warning("Weight gauge update failed (non-blocking): %s", e)


def _load_current_weights() -> dict:
    """Load current weights from config file."""
    import json

    config_path = project_root / "config" / "agent_weights.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("weights", {})
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent performance")

    parser.add_argument("--days", type=int, default=30,
                        help="Lookback period in days (default: 30)")
    parser.add_argument("--min-predictions", type=int, default=20,
                        help="Minimum predictions needed (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be evaluated without storing results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    evaluator = AgentEvaluator()

    print("=" * 60)
    print("MarketSentinel — Agent Performance Evaluation")
    print("=" * 60)
    print()

    if args.dry_run:
        print("DRY RUN MODE - No database changes will be made")
        print()

    print(f"Configuration:")
    print(f"  Lookback Days: {args.days}")
    print(f"  Min Predictions: {args.min_predictions}")
    print()
    print("-" * 60)
    print()

    try:
        results = evaluator.evaluate_agents(
            days=args.days,
            min_predictions=args.min_predictions,
        )

        if not results.get("evaluated"):
            print(f"Evaluation skipped: {results.get('reason', 'unknown')}")
            return 1

        # ── Display results ────────────────────────────────────
        print("Agent Evaluation Complete")
        print()
        print(f"  Total Predictions : {results.get('total_predictions', 0)}")
        print(f"  Period            : {results.get('period_days', 0)} days")
        print(f"  Elapsed           : {results.get('elapsed_seconds', 0):.2f}s")
        print()

        agents = ["signal_agent", "technical_agent", "model_only"]

        for agent in agents:
            if agent not in results:
                continue

            m = results[agent]
            print(f"{agent.replace('_', ' ').title()}:")
            print(f"   Direction Accuracy : {m.get('direction_accuracy', 0):.2%}")
            print(f"   Sharpe Ratio       : {m.get('sharpe', 0):.2f}")
            print(f"   Avg Score          : {m.get('avg_score', 0):.4f}")
            print(f"   Predictions        : {m.get('num_predictions', 0)}")

            if "confidence_calibration" in m:
                print(f"   Conf Calibration   : {m.get('confidence_calibration', 0):.2%}")

            if "mean_absolute_error" in m:
                print(f"   Mean Abs Error     : {m.get('mean_absolute_error', 0):.6f}")
            print()

        # Compare signal vs baseline
        if "signal_agent" in results and "model_only" in results:
            sig_acc = results["signal_agent"].get("direction_accuracy", 0)
            mdl_acc = results["model_only"].get("direction_accuracy", 0)
            improvement = ((sig_acc - mdl_acc) / mdl_acc * 100) if mdl_acc > 0 else 0
            print(f"Signal Agent vs Raw Model: {improvement:+.1f}% accuracy improvement")
            print()

        # ── Issue #28: Update Prometheus gauges ───────────────
        if not args.dry_run:
            _update_prometheus_gauges(results)

            # Also update weight gauges from config
            current_weights = _load_current_weights()
            if current_weights:
                _update_weight_gauges(current_weights)

        if results.get("stored"):
            print("Results stored in agent_performance table")
        else:
            print("Results not stored (dry run or error)")

        return 0

    except KeyboardInterrupt:
        print("\n Interrupted by user")
        return 1

    except Exception as e:
        print(f"\n Error: {e}")
        logger.exception("Agent evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())