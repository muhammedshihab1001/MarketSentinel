"""
MarketSentinel Institutional Training Orchestrator
Production Hardened — Resilient Execution
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
import pandas as pd

from core.schema.feature_schema import get_schema_signature, MODEL_FEATURES
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

TRAINING_TIMEOUT = 60 * 60 * 3


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
# LINEAGE
# ---------------------------------------------------

def snapshot_lineage():

    hasher = hashlib.sha256()

    for root in ["core", "training", "requirements"]:

        if not os.path.exists(root):
            continue

        for path, _, files in os.walk(root):
            for f in sorted(files):

                if f.endswith(".py") or f.endswith(".txt"):
                    with open(os.path.join(path, f), "rb") as fh:
                        hasher.update(fh.read())

    hasher.update(get_schema_signature().encode())

    return hasher.hexdigest()


# ---------------------------------------------------
# VERSION RESOLUTION
# ---------------------------------------------------

def resolve_latest_version(model_name: str) -> str:

    base_dir = f"artifacts/{model_name}"

    if not os.path.exists(base_dir):
        raise RuntimeError(f"{model_name} registry missing.")

    versions = [v for v in os.listdir(base_dir) if v.startswith("v")]

    if not versions:
        raise RuntimeError(f"No versions found for {model_name}")

    return sorted(versions)[-1]


# ---------------------------------------------------
# METADATA
# ---------------------------------------------------

def load_metadata(model_name: str, version: str):

    version_dir = os.path.join(f"artifacts/{model_name}", version)
    metadata_path = os.path.join(version_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"{model_name} metadata missing for {version}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    if metadata.get("metadata_type") != "model":
        raise RuntimeError("Invalid metadata type detected.")

    if metadata.get("schema_signature") != get_schema_signature():
        raise RuntimeError(
            f"{model_name} rejected — schema signature mismatch."
        )

    if "metrics" not in metadata:
        raise RuntimeError("Metadata missing metrics.")

    return metadata


# ---------------------------------------------------
# GOVERNANCE
# ---------------------------------------------------

def governance_check(model_name: str, version: str):

    metadata = load_metadata(model_name, version)
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
        raise RuntimeError(f"{model_name} rejected — sharpe too low")

    if drawdown is not None and drawdown < MAX_DRAWDOWN:
        raise RuntimeError(f"{model_name} rejected — drawdown too severe")

    return metadata


# ---------------------------------------------------
# PROMOTION
# ---------------------------------------------------

def promote_model(model_name: str, version: str):

    base_dir = f"artifacts/{model_name}"

    manifest_path = os.path.join(base_dir, version, "manifest.json")

    if not os.path.exists(manifest_path):
        raise RuntimeError("Manifest missing — cannot promote.")

    ModelRegistry.transition_stage(base_dir, version, "shadow")
    ModelRegistry.transition_stage(base_dir, version, "approved")
    ModelRegistry.promote_to_production(base_dir, version)

    logger.info(f"{model_name} {version} promoted to production.")


# ---------------------------------------------------
# NEW — BASELINE AUTHORITY
# ---------------------------------------------------

def build_drift_baseline_from_xgboost():

    detector = DriftDetector()

    if os.path.exists(detector.BASELINE_PATH):
        logger.info("Drift baseline already exists.")
        return

    logger.info("Creating institutional drift baseline from XGBoost dataset.")

    version = resolve_latest_version("xgboost")

    dataset_path = os.path.join(
        "artifacts",
        "xgboost",
        version,
        "training_dataset.parquet"
    )

    if not os.path.exists(dataset_path):
        raise RuntimeError(
            "XGBoost training dataset missing. Cannot build drift baseline."
        )

    df = pd.read_parquet(dataset_path)

    missing = set(MODEL_FEATURES) - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Baseline creation aborted — dataset missing features: {missing}"
        )

    df = df.loc[:, MODEL_FEATURES]

    if list(df.columns) != MODEL_FEATURES:
        raise RuntimeError("Feature ordering violation during baseline creation.")

    detector.create_baseline(df)


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

def run_step(name: str, module: str):

    logger.info(f"Starting {name} training...")

    start = time.time()

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    process = subprocess.Popen(
        [sys.executable, "-m", module],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True
    )

    try:
        stdout, stderr = process.communicate(timeout=TRAINING_TIMEOUT)

    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(f"{name} training exceeded timeout.")

    if process.returncode != 0:
        logger.error(stderr)
        raise RuntimeError(f"{name} training failed.")

    duration = round(time.time() - start, 2)

    version = resolve_latest_version(name)

    governance_check(name, version)
    promote_model(name, version)

    logger.info(
        f"{name} training completed in {duration}s → version {version}"
    )

    return {
        "model": name,
        "version": version,
        "duration_sec": duration
    }


# ---------------------------------------------------
# MANIFEST
# ---------------------------------------------------

def save_manifest(run_id: str, manifest: dict):

    os.makedirs(RUNS_DIR, exist_ok=True)

    final_path = os.path.join(RUNS_DIR, f"{run_id}.json")
    temp_path = final_path + ".tmp"

    with open(temp_path, "w") as f:
        json.dump(manifest, f, indent=4)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, final_path)


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
    failures = []

    for name, module in PIPELINE_STEPS:

        try:
            results.append(run_step(name, module))

        except Exception as e:

            logger.exception(f"{name} failed.")

            failures.append({
                "model": name,
                "error": str(e)
            })

    if not results:
        logger.error("All training steps failed.")

        manifest = {
            "run_id": run_id,
            "lineage_hash": lineage_hash,
            "reproducibility": reproducibility_stamp(),
            "status": "failed",
            "failures": failures
        }

        save_manifest(run_id, manifest)
        sys.exit(1)

    build_drift_baseline_from_xgboost()

    manifest = {
        "run_id": run_id,
        "lineage_hash": lineage_hash,
        "reproducibility": reproducibility_stamp(),
        "status": "partial_success" if failures else "success",
        "total_runtime_sec": round(time.time() - total_start, 2),
        "results": results,
        "failures": failures
    }

    save_manifest(run_id, manifest)


if __name__ == "__main__":
    main()
