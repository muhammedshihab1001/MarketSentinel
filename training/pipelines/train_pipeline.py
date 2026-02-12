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
import pandas as pd

from training.train_xgboost import main as train_xgb
from core.schema.feature_schema import get_schema_signature
from core.artifacts.metadata_manager import MetadataManager
from core.config.env_loader import init_env
from core.time.market_time import MarketTime
from core.universe.market_universe import MarketUniverse  # VERY IMPORTANT


########################################################
# LOGGING
########################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("marketsentinel.training")

RUNS_DIR = "artifacts/training_runs"

MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40
MAX_REASONABLE_SHARPE = 8.0
MAX_TRAINING_SECONDS = 7200

GLOBAL_SEED = 42


########################################################
# TRUE DETERMINISM
########################################################

def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    try:
        import torch

        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except Exception:
        pass


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
# GIT HASH (FAIL CLOSED)
########################################################

def get_git_commit():

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

    except Exception:
        raise RuntimeError(
            "Training must run inside a git repository."
        )


########################################################
# MANIFEST WRITE
########################################################

def save_manifest(run_id: str, manifest: dict):

    os.makedirs(RUNS_DIR, exist_ok=True)
    _fsync_dir(RUNS_DIR)

    final_path = os.path.join(RUNS_DIR, f"{run_id}.json")
    temp_path = final_path + ".tmp"

    with open(temp_path, "w") as f:
        json.dump(manifest, f, indent=4, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, final_path)
    _fsync_dir(RUNS_DIR)


########################################################
# METRIC VALIDATION
########################################################

def _assert_finite(value, name):
    if value is None or not np.isfinite(value):
        raise RuntimeError(f"Invalid metric produced: {name}")


def validate_metrics(metrics: dict):

    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError("Training returned invalid metrics.")

    required = [
        "avg_sharpe",
        "max_drawdown",
        "profit_factor",
        "final_equity"
    ]

    for key in required:
        _assert_finite(metrics.get(key), key)

    sharpe = metrics["avg_sharpe"]
    drawdown = metrics["max_drawdown"]

    if sharpe > MAX_REASONABLE_SHARPE:
        raise RuntimeError("Sharpe unrealistically high — leakage suspected.")

    if sharpe < MIN_SHARPE:
        raise RuntimeError("Model rejected — Sharpe too low.")

    if drawdown < MAX_DRAWDOWN:
        raise RuntimeError("Model rejected — drawdown too severe.")

    if metrics["final_equity"] <= 0:
        raise RuntimeError("Backtest produced non-positive equity.")


########################################################
# LINEAGE SNAPSHOT (VERY HIGH LEVEL)
########################################################

def build_lineage(start_date, end_date):

    return {
        "schema_signature": get_schema_signature(),
        "training_code_hash": MetadataManager.fingerprint_training_code(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu": platform.processor(),

        # Institutional additions
        "training_window": {
            "start": start_date,
            "end": end_date
        },

        "market_time": MarketTime.snapshot(),

        "universe": MarketUniverse.get_universe()
    }


########################################################
# MAIN
########################################################

def main():

    init_env()
    enforce_determinism()

    ####################################################
    # FREEZE CLOCK (EXTREMELY IMPORTANT)
    ####################################################

    MarketTime.freeze_today(
        datetime.date.today().isoformat()
    )

    start_date, end_date = MarketTime.training_window()

    logger.info(
        "Pipeline training window | start=%s end=%s",
        start_date,
        end_date
    )

    run_id = datetime.datetime.utcnow().strftime(
        "run_%Y_%m_%d_%H%M%S_%f"
    )

    start = time.time()

    try:

        logger.info("Starting XGBoost training...")

        metrics = train_xgb(
            start_date=start_date,
            end_date=end_date
        )

        runtime = time.time() - start

        if runtime > MAX_TRAINING_SECONDS:
            raise RuntimeError("Training exceeded allowed duration.")

        validate_metrics(metrics)

        manifest = {
            "run_id": run_id,
            "status": "success",
            "metrics": metrics,
            "runtime_sec": round(runtime, 2),
            "lineage": build_lineage(start_date, end_date),
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
            "lineage": build_lineage(start_date, end_date),
            "created_utc": datetime.datetime.utcnow().isoformat()
        }

        save_manifest(run_id, manifest)

        raise


if __name__ == "__main__":
    main()
