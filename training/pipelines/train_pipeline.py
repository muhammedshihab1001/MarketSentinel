# ==========================================================
# INSTITUTIONAL TRAINING PIPELINE WRAPPER v2.9
#
# Changes from v2.8:
#   FIX 1: Import ORM models before init_db() so SQLAlchemy's
#           Base.metadata.create_all() can see the table
#           definitions. Without this import, init_db() runs
#           but creates zero tables (Base has nothing registered).
#
#   FIX 2: DataSyncService.sync_universe() now runs BEFORE
#           train_xgb() when data is not present in the DB.
#           Controlled by SKIP_SYNC env var:
#             SKIP_SYNC=0 (default) — always sync before train
#             SKIP_SYNC=1           — skip sync (retrain only)
#           On first run: leave SKIP_SYNC unset or set to 0.
#           On retrain:   set SKIP_SYNC=1 to skip Yahoo fetch.
#
#   FIX 3: DB migration guard — alters schema_signature column
#           from varchar(32) to varchar(64) if needed. Safe to
#           run multiple times (checks current length first).
# ==========================================================

import time
import datetime
import os
import json
import socket
import platform
import random
import subprocess
import tempfile
import uuid
import argparse
import hashlib
import sys

import numpy as np
import pandas as pd

# FIX 1: Import ORM models BEFORE init_db() so Base.metadata
# has all table definitions registered before create_all() runs.
# Without this, init_db() silently creates zero tables.
from core.db.models import OHLCVDaily, ComputedFeature, ModelPrediction  # noqa: F401

from training.train_xgboost import main as train_xgb
from core.schema.feature_schema import (
    get_schema_signature,
    schema_snapshot
)
from core.artifacts.metadata_manager import MetadataManager
from core.config.env_loader import init_env
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse
from core.db.engine import init_db, get_session
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.train_pipeline")

RUNS_DIR = os.path.abspath("artifacts/training_runs")
LOCK_FILE = os.path.join(RUNS_DIR, ".training.lock")

MAX_TRAINING_SECONDS = 7200
GLOBAL_SEED = 42
STRICT_GOVERNANCE = os.getenv("TRAINING_STRICT_GOVERNANCE", "1") == "1"

# FIX 2: SKIP_SYNC=1 skips data sync (use on retrain when data exists)
# SKIP_SYNC=0 (default) always syncs before training
SKIP_SYNC = os.getenv("SKIP_SYNC", "0") == "1"

os.makedirs(RUNS_DIR, exist_ok=True)


# ==========================================================
# DETERMINISM
# ==========================================================

def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)


# ==========================================================
# LOCKING
# ==========================================================

def _acquire_lock():

    if os.path.exists(LOCK_FILE):

        try:
            with open(LOCK_FILE) as f:
                data = json.load(f)
                pid = data.get("pid")
        except Exception:
            pid = None

        age = time.time() - os.path.getmtime(LOCK_FILE)

        if age > 4 * 3600:
            logger.warning("Stale training lock detected — removing")
            try:
                os.remove(LOCK_FILE)
            except FileNotFoundError:
                pass
        else:
            raise RuntimeError(
                f"Training lock detected (pid={pid}) — another training may be running."
            )

    with open(LOCK_FILE, "w") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "created": datetime.datetime.utcnow().isoformat()
            },
            f
        )


def _release_lock():

    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            logger.warning("Failed to remove training lock")


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
            capture_output=True, text=True, timeout=5
        ).stdout.strip()

        if dirty and STRICT_GOVERNANCE:
            raise RuntimeError(
                "Repository is dirty — commit ALL changes before training."
            )

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5
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
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "env_hash": os.getenv("ENV_FINGERPRINT", "none"),
        "process_id": os.getpid()
    }

    canonical = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


# ==========================================================
# DATA WINDOW HASH
# ==========================================================

def build_window_hash(start_date, end_date):

    payload = {"start": start_date, "end": end_date}
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
        "training_window": {"start": start_date, "end": end_date},
        "window_hash": build_window_hash(start_date, end_date),
        "training_universe": universe_snapshot,
        "universe_hash": universe_hash
    }

    canonical = json.dumps(lineage_payload, sort_keys=True).encode()
    lineage_payload["lineage_hash"] = hashlib.sha256(canonical).hexdigest()

    return lineage_payload


# ==========================================================
# PROVIDER HEALTH
# ==========================================================

def capture_provider_health():

    try:
        from sqlalchemy import text
        with get_session() as session:
            session.execute(text("SELECT 1"))
        return {"status": "db_healthy", "source": "postgresql"}
    except Exception as e:
        logger.warning("DB health check failed | error=%s", e)
        return {"status": "db_unavailable", "error": str(e)}


# ==========================================================
# MANIFEST SAVE
# ==========================================================

def save_manifest(run_id: str, manifest: dict):

    final_path = os.path.join(RUNS_DIR, f"{run_id}.json")

    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=RUNS_DIR,
        suffix=".tmp", encoding="utf-8"
    ) as tmp:
        json.dump(manifest, tmp, indent=4, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name

    os.replace(temp_name, final_path)


# ==========================================================
# DB MIGRATION — schema_signature varchar(32) → varchar(64)
# FIX 3: sha256 hex is 64 chars. Old column was 32. Every
# INSERT to model_predictions failed with truncation error.
# Safe to run multiple times — checks current length first.
# ==========================================================

def _run_db_migrations():
    """
    Apply any pending DB schema migrations.
    Safe to run on every startup — idempotent checks before altering.
    """
    try:
        from sqlalchemy import text

        with get_session() as session:

            # Check current column length for schema_signature
            result = session.execute(text("""
                SELECT character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'model_predictions'
                AND column_name = 'schema_signature'
            """)).fetchone()

            if result and result[0] and int(result[0]) < 64:
                session.execute(text("""
                    ALTER TABLE model_predictions
                    ALTER COLUMN schema_signature TYPE varchar(64)
                """))
                logger.info(
                    "Migration applied: schema_signature varchar(32) → varchar(64)"
                )
            else:
                logger.info("DB migrations: schema_signature already correct.")

    except Exception as e:
        # Non-blocking — migration failure must not stop training
        logger.warning("DB migration failed (non-blocking) | error=%s", e)


# ==========================================================
# DATA SYNC BEFORE TRAINING
# FIX 2: Runs sync before training so DB has data.
# Skipped if SKIP_SYNC=1 (set on retrain when data exists).
# ==========================================================

def _sync_data_if_needed():
    """
    Sync market data from Yahoo Finance → PostgreSQL.

    Called before training to ensure DB has data.
    Skipped if SKIP_SYNC=1 env var is set.

    First run:   SKIP_SYNC=0 (default) — fetches 2 years per ticker
    Retrain:     SKIP_SYNC=1           — skips Yahoo, uses existing data
    """

    if SKIP_SYNC:
        logger.info(
            "Data sync skipped (SKIP_SYNC=1). "
            "Using existing DB data for training."
        )
        return

    logger.info(
        "Starting data sync before training "
        "(set SKIP_SYNC=1 to skip on retrain)"
    )

    try:
        from core.data.data_sync import DataSyncService

        svc = DataSyncService()
        report = svc.sync_universe()

        logger.info(
            "Pre-training data sync complete | "
            "synced=%d skipped=%d errors=%d rows_inserted=%d",
            report.get("synced", 0),
            report.get("skipped", 0),
            report.get("errors", 0),
            report.get("total_rows_inserted", 0),
        )

        # Warn if too many tickers failed — training may have insufficient data
        error_count = report.get("errors", 0)
        total = report.get("total_tickers", 1)

        if error_count > total * 0.5:
            logger.warning(
                "More than 50%% of tickers failed sync (%d/%d). "
                "Training data may be insufficient. "
                "Check Yahoo Finance connectivity.",
                error_count, total,
            )

    except Exception as exc:
        # Non-blocking — if sync fails, training uses whatever is in DB
        logger.warning(
            "Pre-training sync failed (non-blocking) | error=%s | "
            "Training will use existing DB data.",
            exc,
        )


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main(create_baseline=False, promote_baseline=False):

    init_env()
    enforce_determinism()

    # ── Step 1: Init DB (FIX 1 + FIX 3) ─────────────────
    # Models imported at top of file — Base.metadata is populated.
    # init_db() will now correctly create all three tables.
    logger.info("Initialising database connection | function=main")

    try:
        init_db()
        logger.info("Database ready | function=main")
    except Exception as e:
        logger.warning(
            "DB init failed — training will use whatever data is available | error=%s", e
        )

    # ── Step 2: Run DB migrations ─────────────────────────
    _run_db_migrations()

    # ── Step 3: Sync data before training (FIX 2) ─────────
    _sync_data_if_needed()

    # ── Step 4: Acquire lock + run training ───────────────
    _acquire_lock()

    today = MarketTime.today().isoformat()
    MarketTime.freeze_today(today)

    start_date, end_date = MarketTime.window_for("xgboost")

    logger.info(
        "Training window | start=%s | end=%s | function=main",
        start_date, end_date,
    )

    schema_signature = get_schema_signature()

    if not schema_signature:
        raise RuntimeError("Schema signature invalid.")

    run_id = (
        datetime.datetime.utcnow().strftime("run_%Y_%m_%d_%H%M%S_%f")
        + "_" + uuid.uuid4().hex[:6]
    )

    logger.info("Training run | run_id=%s | function=main", run_id)

    start = time.time()

    try:

        metrics = train_xgb(
            start_date=start_date,
            end_date=end_date,
            create_baseline=create_baseline,
            promote_baseline=promote_baseline,
        )

        if not isinstance(metrics, dict):
            raise RuntimeError("Training did not return metrics dictionary.")

        runtime = max(0.0, time.time() - start)

        if runtime > MAX_TRAINING_SECONDS:
            raise RuntimeError("Training exceeded allowed duration.")

        manifest = {
            "run_id": run_id,
            "status": "success",
            "schema_signature": schema_signature,
            "training_window": {"start": start_date, "end": end_date},
            "metrics": metrics,
            "runtime_sec": round(runtime, 2),
            "provider_health": capture_provider_health(),
            "lineage": build_lineage(start_date, end_date),
            "created_utc": datetime.datetime.utcnow().isoformat(),
        }

        save_manifest(run_id, manifest)

        logger.info(
            "Training pipeline finished successfully | run_id=%s | function=main",
            run_id,
        )

    except Exception as exc:

        logger.exception("Training failed | run_id=%s | function=main", run_id)

        manifest = {
            "run_id": run_id,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "runtime_sec": round(max(0.0, time.time() - start), 2),
            "provider_health": capture_provider_health(),
            "lineage": build_lineage(start_date, end_date),
            "created_utc": datetime.datetime.utcnow().isoformat(),
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
        promote_baseline=args.promote_baseline,
    )
