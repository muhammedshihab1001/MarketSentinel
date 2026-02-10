"""
MarketSentinel Institutional Training Orchestrator

Promotion-safe release pipeline.
Cross-platform.
Registry-aware.
"""

import subprocess
import sys
import time
import json
import datetime
import os
import hashlib
import platform
import logging

from core.schema.feature_schema import get_schema_signature
from core.monitoring.drift_detector import DriftDetector
from core.artifacts.model_registry import ModelRegistry


logger = logging.getLogger("marketsentinel.training")


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
# TRUE LINEAGE SNAPSHOT
# ---------------------------------------------------

def snapshot_lineage():

    hasher = hashlib.sha256()

    paths = [
        "core/features",
        "core/schema",
        "training",
        "requirements"
    ]

    for root in paths:

        if not os.path.exists(root):
            continue

        for path, _, files in os.walk(root):

            for f in sorted(files):

                if f.endswith(".py") or f.endswith(".txt"):
                    full = os.path.join(path, f)

                    with open(full, "rb") as fh:
                        hasher.update(fh.read())

    hasher.update(get_schema_signature().encode())

    return hasher.hexdigest()


# ---------------------------------------------------
# METADATA
# ---------------------------------------------------

def load_metadata(model_name: str):

    version = ModelRegistry.get_latest_version(
        f"artifacts/{model_name}"
    )

    path = os.path.join(
        f"artifacts/{model_name}",
        version,
        "metadata.json"
    )

    if not os.path.exists(path):
        raise RuntimeError(f"{model_name} metadata missing.")

    with open(path) as f:
        metadata = json.load(f)

    if "metrics" not in metadata:
        raise RuntimeError(f"{model_name} metadata missing metrics.")

    if "schema_signature" not in metadata:
        raise RuntimeError(f"{model_name} missing schema signature.")

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

    return metadata


# ---------------------------------------------------
# REGISTRY SAFETY (FIXED)
# ---------------------------------------------------

def validate_registry(model_name: str):

    base_dir = f"artifacts/{model_name}"

    version = ModelRegistry.get_latest_version(base_dir)

    version_dir = os.path.join(base_dir, version)

    manifest = os.path.join(version_dir, "manifest.json")
    metadata = os.path.join(version_dir, "metadata.json")

    if not os.path.exists(manifest):
        raise RuntimeError("Registry manifest missing.")

    if not os.path.exists(metadata):
        raise RuntimeError("Registry metadata missing.")

    with open(manifest) as f:
        data = json.load(f)

    if data.get("stage") != "production":
        raise RuntimeError(
            f"{model_name} not promoted to production."
        )


# ---------------------------------------------------
# DRIFT BASELINE
# ---------------------------------------------------

def ensure_drift_baseline():

    detector = DriftDetector()

    if os.path.exists(detector.BASELINE_PATH):
        return

    raise RuntimeError(
        "Drift baseline missing. Training must generate baseline."
    )


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

def run_step(name: str, module: str):

    logger.info(f"Starting {name} training...")

    start = time.time()

    subprocess.run(
        [sys.executable, "-m", module],
        check=True
    )

    duration = round(time.time() - start, 2)

    metadata = governance_check(name)
    validate_registry(name)

    return {
        "model": name,
        "metrics": metadata["metrics"],
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

    logger.info(f"Training manifest saved -> {path}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    run_id = datetime.datetime.utcnow().strftime(
        "run_%Y_%m_%d_%H%M%S"
    )

    total_start = time.time()

    lineage_hash = snapshot_lineage()

    results = []

    try:

        for name, module in PIPELINE_STEPS:

            result = run_step(name, module)
            results.append(result)

        ensure_drift_baseline()

    except Exception as e:

        logger.exception("Training run failed.")

        total_time = round(time.time() - total_start, 2)

        manifest = {
            "run_id": run_id,
            "lineage_hash": lineage_hash,
            "reproducibility": reproducibility_stamp(),
            "status": "failed",
            "error": str(e),
            "results": results
        }

        save_manifest(run_id, manifest)

        sys.exit(1)

    total_time = round(time.time() - total_start, 2)

    manifest = {
        "run_id": run_id,
        "lineage_hash": lineage_hash,
        "reproducibility": reproducibility_stamp(),
        "status": "success",
        "total_runtime_sec": total_time,
        "results": results
    }

    save_manifest(run_id, manifest)


if __name__ == "__main__":
    main()
