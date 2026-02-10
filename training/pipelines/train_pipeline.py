"""
MarketSentinel Institutional Training Orchestrator

Release-grade training pipeline.

Guarantees:
- dataset snapshot
- reproducibility stamp
- governance gates BEFORE promotion reliance
- registry validation
- promotion report
"""

import subprocess
import sys
import time
import json
import datetime
import os
import hashlib
import platform


PIPELINE_STEPS = [
    ("xgboost", "training.train_xgboost"),
    ("lstm", "training.train_lstm"),
    ("prophet", "training.train_prophet"),
]

RUNS_DIR = "artifacts/training_runs"


MIN_ACCURACY = 0.50
MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40


# ---------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------

def reproducibility_stamp():

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.datetime.utcnow().isoformat()
    }


# ---------------------------------------------------
# DATASET SNAPSHOT
# ---------------------------------------------------

def snapshot_artifacts():

    """
    Hash critical artifacts to create a lineage anchor.

    This avoids requiring a full data lake while still
    providing institutional reproducibility.
    """

    hasher = hashlib.sha256()

    artifact_roots = [
        "data/features",
        "artifacts/drift"
    ]

    for root in artifact_roots:

        if not os.path.exists(root):
            continue

        for path, _, files in os.walk(root):

            for f in sorted(files):

                full = os.path.join(path, f)

                try:
                    with open(full, "rb") as fh:
                        hasher.update(fh.read())
                except Exception:
                    continue

    return hasher.hexdigest()


# ---------------------------------------------------
# METADATA SAFETY
# ---------------------------------------------------

def load_metadata(model_name: str):

    path = f"artifacts/{model_name}/latest/metadata.json"

    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing metadata for {model_name}. "
            "Registry promotion likely failed."
        )

    with open(path) as f:
        metadata = json.load(f)

    if "metrics" not in metadata:
        raise RuntimeError(
            f"{model_name} metadata missing metrics block."
        )

    return metadata


# ---------------------------------------------------
# GOVERNANCE
# ---------------------------------------------------

def governance_check(model_name: str):

    metadata = load_metadata(model_name)

    metrics = metadata["metrics"]

    if model_name == "xgboost":

        acc = metrics.get("accuracy")

        if acc is None or acc < MIN_ACCURACY:
            raise RuntimeError(
                f"{model_name} rejected — accuracy below threshold"
            )

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
# REGISTRY VALIDATION
# ---------------------------------------------------

def validate_registry_pointer(model_name: str):

    latest = f"artifacts/{model_name}/latest"

    if not os.path.islink(latest):
        raise RuntimeError(
            f"{model_name} latest pointer missing."
        )

    resolved = os.readlink(latest)

    version_dir = os.path.join(
        f"artifacts/{model_name}",
        resolved
    )

    manifest = os.path.join(version_dir, "manifest.json")

    if not os.path.exists(manifest):
        raise RuntimeError(
            f"{model_name} manifest missing — registry corrupted."
        )


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

def run_step(name: str, module: str):

    start = time.time()

    subprocess.run(
        [sys.executable, "-m", module],
        check=True
    )

    duration = round(time.time() - start, 2)

    governance_check(name)
    validate_registry_pointer(name)

    return {
        "model": name,
        "status": "success",
        "duration_sec": duration
    }


# ---------------------------------------------------
# MANIFEST
# ---------------------------------------------------

def save_manifest(run_id: str, manifest: dict):

    os.makedirs(RUNS_DIR, exist_ok=True)

    path = os.path.join(RUNS_DIR, f"{run_id}.json")

    with open(path, "w") as f:
        json.dump(manifest, f, indent=4)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    run_id = datetime.datetime.utcnow().strftime(
        "run_%Y_%m_%d_%H%M%S"
    )

    total_start = time.time()

    dataset_hash = snapshot_artifacts()

    results = []

    try:

        for name, module in PIPELINE_STEPS:

            result = run_step(name, module)
            results.append(result)

    except Exception as e:

        results.append({
            "status": "failed",
            "error": str(e)
        })

        total_time = round(time.time() - total_start, 2)

        manifest = {
            "run_id": run_id,
            "dataset_snapshot": dataset_hash,
            "reproducibility": reproducibility_stamp(),
            "total_runtime_sec": total_time,
            "results": results
        }

        save_manifest(run_id, manifest)

        sys.exit(1)

    total_time = round(time.time() - total_start, 2)

    manifest = {
        "run_id": run_id,
        "dataset_snapshot": dataset_hash,
        "reproducibility": reproducibility_stamp(),
        "total_runtime_sec": total_time,
        "results": results
    }

    save_manifest(run_id, manifest)


# ---------------------------------------------------

if __name__ == "__main__":
    main()
