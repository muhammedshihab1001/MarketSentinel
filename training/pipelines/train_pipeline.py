import time
import datetime
import os
import json
import logging
import socket
import platform
import random
import subprocess
import tempfile
import uuid

import numpy as np
import pandas as pd

from training.train_xgboost import main as train_xgb
from core.schema.feature_schema import get_schema_signature
from core.artifacts.metadata_manager import MetadataManager
from core.config.env_loader import init_env
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse


logger = logging.getLogger(__name__)

RUNS_DIR = os.path.abspath("artifacts/training_runs")
LOCK_FILE = os.path.join(RUNS_DIR, ".training.lock")

MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40
MAX_REASONABLE_SHARPE = 8.0
MAX_PROFIT_FACTOR = 10.0
MAX_TRAINING_SECONDS = 7200

GLOBAL_SEED = 42


def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)


def _acquire_lock():

    os.makedirs(RUNS_DIR, exist_ok=True)

    if os.path.exists(LOCK_FILE):
        raise RuntimeError(
            "Training lock detected — another training may be running."
        )

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

    return True


def _release_lock():

    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)


def _fsync_dir(directory):

    if os.name == "nt":
        return

    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def get_git_commit():

    if not os.path.exists(".git"):
        raise RuntimeError(
            "Training must run inside a FULL git repository."
        )

    try:

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        if dirty:
            raise RuntimeError(
                "Repository is dirty — commit ALL changes before training."
            )

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        if len(commit) != 40:
            raise RuntimeError("Invalid git commit hash.")

        return commit

    except subprocess.CalledProcessError:
        raise RuntimeError(
            "Unable to resolve git commit — refusing training."
        )


def save_manifest(run_id: str, manifest: dict):

    os.makedirs(RUNS_DIR, exist_ok=True)

    final_path = os.path.join(RUNS_DIR, f"{run_id}.json")

    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=RUNS_DIR,
        suffix=".tmp"
    ) as tmp:

        json.dump(manifest, tmp, indent=4, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())

        temp_name = tmp.name

    os.replace(temp_name, final_path)
    _fsync_dir(RUNS_DIR)

    # verify readability
    with open(final_path, "r") as f:
        json.load(f)


def _assert_finite(value, name):
    if value is None or not np.isfinite(value):
        raise RuntimeError(f"Invalid metric produced: {name}")


def validate_metrics(metrics: dict):

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
    profit_factor = metrics["profit_factor"]

    if sharpe > MAX_REASONABLE_SHARPE:
        raise RuntimeError("Sharpe unrealistically high — leakage suspected.")

    if sharpe < MIN_SHARPE:
        raise RuntimeError("Model rejected — Sharpe too low.")

    if drawdown < MAX_DRAWDOWN:
        raise RuntimeError("Model rejected — drawdown too severe.")

    if profit_factor > MAX_PROFIT_FACTOR:
        raise RuntimeError("Profit factor unrealistic — leakage suspected.")

    if metrics["final_equity"] <= 0:
        raise RuntimeError("Backtest produced non-positive equity.")

    win_rate = metrics.get("win_rate")
    if win_rate is not None and win_rate > 0.80:
        logger.warning("Win rate extremely high — investigate leakage.")


def build_lineage(start_date, end_date):

    universe_snapshot = MarketUniverse.snapshot()

    lineage_payload = {
        "schema_signature": get_schema_signature(),
        "training_code_hash": MetadataManager.fingerprint_training_code(),
        "git_commit": get_git_commit(),

        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),

        "env": {
            "cpu_count": os.cpu_count(),
            "machine": platform.machine()
        },

        "training_window": {
            "start": start_date,
            "end": end_date
        },

        "market_time_snapshot": MarketTime.snapshot_for("xgboost"),
        "training_universe": universe_snapshot
    }

    canonical = json.dumps(
        lineage_payload,
        sort_keys=True
    ).encode()

    lineage_payload["lineage_hash"] = (
        MetadataManager.hash_list([canonical.hex()])
    )

    return lineage_payload


def main():

    init_env()
    enforce_determinism()

    _acquire_lock()

    today = MarketTime.today().isoformat()
    MarketTime.freeze_today(today)

    start_date, end_date = MarketTime.window_for("xgboost")

    logger.info(
        "Pipeline training window | %s -> %s",
        start_date,
        end_date
    )

    run_id = (
        datetime.datetime.utcnow().strftime("run_%Y_%m_%d_%H%M%S_%f")
        + "_" + uuid.uuid4().hex[:6]
    )

    start = time.time()

    try:

        logger.info("Starting institutional XGBoost training...")

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
            "lineage": build_lineage(
                start_date,
                end_date
            ),
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
            "lineage": build_lineage(
                start_date,
                end_date
            ),
            "created_utc": datetime.datetime.utcnow().isoformat()
        }

        save_manifest(run_id, manifest)

        raise

    finally:
        _release_lock()


if __name__ == "__main__":
    main()
