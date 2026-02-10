"""
MarketSentinel Institutional Training Orchestrator

NOW WITH MODEL GOVERNANCE.

Guarantees:
✅ training success
✅ metadata validation
✅ performance gates
✅ deployment safety
"""

import subprocess
import sys
import time
import json
import datetime
import os


PIPELINE_STEPS = [
    ("xgboost", "training.train_xgboost"),
    ("lstm", "training.train_lstm"),
    ("prophet", "training.train_prophet"),
]

RUNS_DIR = "artifacts/training_runs"

# ---------------------------------------------------
# 🔥 GOVERNANCE THRESHOLDS
# ---------------------------------------------------

MIN_ACCURACY = 0.50
MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40   # -40%


# ---------------------------------------------------

def load_metadata(model_name: str):

    path = f"artifacts/{model_name}/latest/metadata.json"

    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing metadata for {model_name}. "
            "Model was not properly registered."
        )

    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------

def governance_check(model_name: str):

    metadata = load_metadata(model_name)

    metrics = metadata.get("metrics", {})

    # XGBoost accuracy gate
    if model_name == "xgboost":

        acc = metrics.get("accuracy")

        if acc is None or acc < MIN_ACCURACY:
            raise RuntimeError(
                f"{model_name} rejected — accuracy below threshold"
            )

    # Walk-forward metrics (if present)
    sharpe = metrics.get("avg_sharpe")
    drawdown = metrics.get("max_drawdown")

    if sharpe is not None and sharpe < MIN_SHARPE:
        raise RuntimeError(
            f"{model_name} rejected — sharpe too low"
        )

    if drawdown is not None and drawdown < MAX_DRAWDOWN:
        raise RuntimeError(
            f"{model_name} rejected — drawdown too severe"
        )


# ---------------------------------------------------

def run_step(name: str, module: str):

    print("\n===================================")
    print(f" 🚀 Starting {name.upper()} training")
    print("===================================\n")

    start = time.time()

    try:

        subprocess.run(
            [sys.executable, "-m", module],
            check=True
        )

        duration = round(time.time() - start, 2)

        # 🔥 GOVERNANCE CHECK
        governance_check(name)

        print(f"\n✅ {name.upper()} PASSED governance in {duration}s\n")

        return {
            "model": name,
            "status": "success",
            "duration_sec": duration
        }

    except Exception as e:

        duration = round(time.time() - start, 2)

        print(f"\n❌ {name.upper()} FAILED after {duration}s\n")

        return {
            "model": name,
            "status": "failed",
            "duration_sec": duration,
            "error": str(e)
        }


# ---------------------------------------------------

def save_manifest(run_id: str, manifest: dict):

    os.makedirs(RUNS_DIR, exist_ok=True)

    path = os.path.join(RUNS_DIR, f"{run_id}.json")

    with open(path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n📄 Training manifest saved → {path}\n")


# ---------------------------------------------------

def main():

    run_id = datetime.datetime.utcnow().strftime(
        "run_%Y_%m_%d_%H%M%S"
    )

    total_start = time.time()

    print("\n########################################")
    print("   MARKET SENTINEL — TRAINING RUN")
    print(f"   RUN ID: {run_id}")
    print("########################################\n")

    results = []

    for name, module in PIPELINE_STEPS:

        result = run_step(name, module)
        results.append(result)

        if result["status"] == "failed":
            break

    total_time = round(time.time() - total_start, 2)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "total_runtime_sec": total_time,
        "results": results
    }

    save_manifest(run_id, manifest)

    failures = [r for r in results if r["status"] == "failed"]

    if failures:

        print("\n########################################")
        print(" ❌ TRAINING RUN FAILED")
        print("########################################\n")

        sys.exit(1)

    print("\n########################################")
    print(" ✅ ALL MODELS PASSED GOVERNANCE")
    print(f" ⏱ Total runtime: {total_time}s")
    print("########################################\n")


# ---------------------------------------------------

if __name__ == "__main__":
    main()
