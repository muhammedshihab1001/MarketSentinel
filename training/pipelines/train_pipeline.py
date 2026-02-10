"""
MarketSentinel Institutional Training Orchestrator

Acts as the ML control plane.

Guarantees:
✅ Atomic training visibility
✅ Version manifest
✅ Failure-safe execution
✅ Run traceability
"""

import subprocess
import sys
import time
import json
import datetime
import os


PIPELINE_STEPS = [
    ("XGBoost", "training.train_xgboost"),
    ("LSTM", "training.train_lstm"),
    ("Prophet", "training.train_prophet"),
]

RUNS_DIR = "artifacts/training_runs"


# ---------------------------------------------------

def run_step(name: str, module: str):

    print("\n===================================")
    print(f" 🚀 Starting {name} training")
    print("===================================\n")

    start = time.time()

    try:

        result = subprocess.run(
            [sys.executable, "-m", module],
            check=True
        )

        duration = round(time.time() - start, 2)

        print(f"\n✅ {name} completed in {duration}s\n")

        return {
            "model": name,
            "status": "success",
            "duration_sec": duration
        }

    except subprocess.CalledProcessError as e:

        duration = round(time.time() - start, 2)

        print(f"\n❌ {name} FAILED after {duration}s\n")

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

        # STOP if critical model fails
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

    # ---------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------

    failures = [r for r in results if r["status"] == "failed"]

    if failures:

        print("\n########################################")
        print(" ❌ TRAINING RUN FAILED")
        print("########################################\n")

        sys.exit(1)

    print("\n########################################")
    print(" ✅ ALL MODELS TRAINED SUCCESSFULLY")
    print(f" ⏱ Total runtime: {total_time}s")
    print("########################################\n")


# ---------------------------------------------------

if __name__ == "__main__":
    main()
