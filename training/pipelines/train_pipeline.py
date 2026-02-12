import time
import datetime
import os
import json
import logging
import socket
import platform
import random
import subprocess

import numpy as np

from training.train_xgboost import main as train_xgb
from core.schema.feature_schema import get_schema_signature
from core.artifacts.metadata_manager import MetadataManager


logger = logging.getLogger("marketsentinel.training")

RUNS_DIR = "artifacts/training_runs"

MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40
MAX_REASONABLE_SHARPE = 8.0   # institutional sanity guard

GLOBAL_SEED = 42


########################################################
# TRUE DETERMINISM
########################################################

def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

    #  BLAS / OpenMP stability
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # future-safe
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)


########################################################
# FSYNC DIRECTORY
########################################################

def _fsync_dir(directory):

    if os.name == "nt":
        return

    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


########################################################
# GIT HASH (BEST EFFORT)
########################################################

def get_git_commit():

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


########################################################
# MANIFEST WRITE
########################################################

def save_manifest(run_id: str, manifest: dict):

    os.makedirs(RUNS_DIR, exist_ok=True)

    final_path = os.path.join(RUNS_DIR, f"{run_id}.json")
    temp_path = final_path + ".tmp"

    with open(temp_path, "w") as f:
        json.dump(manifest, f, indent=4)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, final_path)

    _fsync_dir(RUNS_DIR)


########################################################
# METRIC VALIDATION
########################################################

def validate_metrics(metrics: dict):

    if not isinstance(metrics, dict):
        raise RuntimeError("Training returned invalid metrics.")

    sharpe = metrics.get("avg_sharpe")
    drawdown = metrics.get("max_drawdown")

    if sharpe is None:
        raise RuntimeError("Training failed — missing sharpe.")

    if not np.isfinite(sharpe):
        raise RuntimeError("Invalid sharpe produced.")

    if sharpe > MAX_REASONABLE_SHARPE:
        raise RuntimeError(
            "Sharpe unrealistically high — likely leakage."
        )

    if drawdown is not None and not np.isfinite(drawdown):
        raise RuntimeError("Invalid drawdown produced.")

    if sharpe < MIN_SHARPE:
        raise RuntimeError("Model rejected — sharpe too low.")

    if drawdown is not None and drawdown < MAX_DRAWDOWN:
        raise RuntimeError("Model rejected — drawdown too severe.")


########################################################
# LINEAGE SNAPSHOT
########################################################

def build_lineage():

    return {
        "schema_signature": get_schema_signature(),
        "training_code_hash": MetadataManager.fingerprint_training_code(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
    }


########################################################
# MAIN
########################################################

def main():

    enforce_determinism()

    #  microsecond-safe run id
    run_id = datetime.datetime.utcnow().strftime(
        "run_%Y_%m_%d_%H%M%S_%f"
    )

    start = time.time()

    try:

        logger.info("Starting XGBoost training...")

        metrics = train_xgb()

        validate_metrics(metrics)

        manifest = {
            "run_id": run_id,
            "status": "success",
            "metrics": metrics,
            "runtime_sec": round(time.time() - start, 2),
            "lineage": build_lineage(),
            "created_utc": datetime.datetime.utcnow().isoformat()
        }

        save_manifest(run_id, manifest)

        logger.info("Training completed successfully.")

    except Exception as exc:

        logger.exception("Training failed.")

        manifest = {
            "run_id": run_id,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "runtime_sec": round(time.time() - start, 2),
            "lineage": build_lineage(),
            "created_utc": datetime.datetime.utcnow().isoformat()
        }

        save_manifest(run_id, manifest)

        raise


if __name__ == "__main__":
    main()
