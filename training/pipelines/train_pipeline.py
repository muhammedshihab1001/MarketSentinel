# ==========================================================
# INSTITUTIONAL TRAINING PIPELINE WRAPPER v2.5
# Governance Hardened + Hybrid Multi-Agent Compatible
# ==========================================================

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
import argparse
import hashlib

import numpy as np
import pandas as pd

from training.train_xgboost import main as train_xgb
from core.schema.feature_schema import (
    get_schema_signature,
    schema_snapshot
)
from core.artifacts.metadata_manager import MetadataManager
from core.config.env_loader import init_env
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse
from core.data.providers.market.router import MarketProviderRouter

logger = logging.getLogger(__name__)

RUNS_DIR = os.path.abspath("artifacts/training_runs")
LOCK_FILE = os.path.join(RUNS_DIR, ".training.lock")

MAX_TRAINING_SECONDS = 7200
GLOBAL_SEED = 42
STRICT_GOVERNANCE = os.getenv("TRAINING_STRICT_GOVERNANCE", "1") == "1"


# ==========================================================
# DETERMINISM
# ==========================================================

def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # optional deterministic flags (safe for portfolio project)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)


# ==========================================================
# LOCKING
# ==========================================================

def _acquire_lock():

    os.makedirs(RUNS_DIR, exist_ok=True)

    if os.path.exists(LOCK_FILE):

        try:
            with open(LOCK_FILE) as f:
                pid = int(f.read().strip())
        except Exception:
            pid = None

        age = time.time() - os.path.getmtime(LOCK_FILE)

        if age > 4 * 3600:
            logger.warning("Stale training lock detected — removing.")
            os.remove(LOCK_FILE)

        else:
            raise RuntimeError(
                f"Training lock detected (pid={pid}) — another training may be running."
            )

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))


def _release_lock():

    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            logger.warning("Failed to remove training lock.")


# ==========================================================
# GIT SAFETY
# ==========================================================

def get_git_commit():

    if not os.path.exists(".git"):
        logger.info("Git repository not present (container mode).")
        return "NO_GIT"

    try:

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True
        ).stdout.strip()

        if dirty and STRICT_GOVERNANCE:
            raise RuntimeError(
                "Repository is dirty — commit ALL changes before training."
            )

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return commit

    except Exception:
        logger.warning("Git commit unavailable.")
        return "GIT_UNAVAILABLE"


# ==========================================================
# ENV + HARDWARE FINGERPRINT
# ==========================================================

def build_environment_fingerprint():

    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "env_hash": os.getenv("ENV_FINGERPRINT"),
    }

    canonical = json.dumps(payload, sort_keys=True).encode()

    return hashlib.sha256(canonical).hexdigest()


# ==========================================================
# DATA WINDOW HASH
# ==========================================================

def build_window_hash(start_date, end_date):

    payload = {
        "start": start_date,
        "end": end_date
    }

    canonical = json.dumps(payload, sort_keys=True).encode()

    return hashlib.sha256(canonical).hexdigest()


# ==========================================================
# LINEAGE
# ==========================================================

def build_lineage(start_date, end_date):

    universe_snapshot = MarketUniverse.snapshot()
    universe_hash = MarketUniverse.fingerprint()

    training_code_hash = MetadataManager.fingerprint_training_code()

    lineage_payload = {
        "schema_signature": get_schema_signature(),
        "schema_snapshot": schema_snapshot(),
        "training_code_hash": training_code_hash,
        "git_commit": get_git_commit(),
        "environment_fingerprint": build_environment_fingerprint(),
        "training_window": {
            "start": start_date,
            "end": end_date
        },
        "window_hash": build_window_hash(start_date, end_date),
        "training_universe": universe_snapshot,
        "universe_hash": universe_hash
    }

    canonical = json.dumps(lineage_payload, sort_keys=True).encode()

    lineage_payload["lineage_hash"] = hashlib.sha256(canonical).hexdigest()

    return lineage_payload


# ==========================================================
# PROVIDER HEALTH SNAPSHOT
# ==========================================================

def capture_provider_health():

    try:

        router = MarketProviderRouter()

        if hasattr(router, "provider_health"):
            return router.provider_health()

        return {"status": "provider_health_not_available"}

    except Exception as e:

        logger.warning("Failed to capture provider health: %s", e)

        return {}


# ==========================================================
# SAFE MANIFEST SAVE
# ==========================================================

def save_manifest(run_id: str, manifest: dict):

    os.makedirs(RUNS_DIR, exist_ok=True)

    final_path = os.path.join(RUNS_DIR, f"{run_id}.json")

    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=RUNS_DIR,
        suffix=".tmp",
        encoding="utf-8"
    ) as tmp:

        json.dump(manifest, tmp, indent=4, sort_keys=True)

        tmp.flush()
        os.fsync(tmp.fileno())

        temp_name = tmp.name

    os.replace(temp_name, final_path)


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main(create_baseline=False, promote_baseline=False):

    init_env()

    enforce_determinism()

    _acquire_lock()

    today = MarketTime.today().isoformat()

    MarketTime.freeze_today(today)

    start_date, end_date = MarketTime.window_for("xgboost")

    logger.info("Training window | %s -> %s", start_date, end_date)

    current_schema = get_schema_signature()

    if not current_schema:
        raise RuntimeError("Schema signature invalid.")

    run_id = (
        datetime.datetime.utcnow().strftime("run_%Y_%m_%d_%H%M%S_%f")
        + "_" + uuid.uuid4().hex[:6]
    )

    logger.info("Training run_id=%s", run_id)

    start = time.time()

    try:

        metrics = train_xgb(
            start_date=start_date,
            end_date=end_date,
            create_baseline=create_baseline,
            promote_baseline=promote_baseline
        )

        runtime = max(0.0, time.time() - start)

        if runtime > MAX_TRAINING_SECONDS:
            raise RuntimeError("Training exceeded allowed duration.")

        manifest = {
            "run_id": run_id,
            "status": "success",
            "metrics": metrics,
            "runtime_sec": round(runtime, 2),
            "provider_health": capture_provider_health(),
            "lineage": build_lineage(start_date, end_date),
            "created_utc": datetime.datetime.utcnow().isoformat()
        }

        save_manifest(run_id, manifest)

        logger.info("Training pipeline finished successfully.")

    except Exception as exc:

        logger.exception("Training failed.")

        manifest = {
            "run_id": run_id,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "runtime_sec": round(max(0.0, time.time() - start), 2),
            "provider_health": capture_provider_health(),
            "lineage": build_lineage(start_date, end_date),
            "created_utc": datetime.datetime.utcnow().isoformat()
        }

        save_manifest(run_id, manifest)

        raise

    finally:

        _release_lock()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--create-baseline", action="store_true")
    parser.add_argument("--promote-baseline", action="store_true")

    args = parser.parse_args()

    main(
        create_baseline=args.create_baseline,
        promote_baseline=args.promote_baseline
    )