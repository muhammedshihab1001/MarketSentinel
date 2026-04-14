#!/bin/sh
# ==========================================================
# MarketSentinel — Training Container Entrypoint v1.1
#
# Changes from v1.0:
#   FIX: SKIP_SYNC env var was checked here for logging
#        but Python code reads SKIP_DATA_SYNC (different name).
#        Setting SKIP_SYNC=1 had no effect on Python sync.
#        Fix: export SKIP_DATA_SYNC=$SKIP_SYNC so both work.
#        SKIP_DATA_SYNC=1 still works directly (backward compat).
#
# Automates the full training boot sequence in one shot:
#   1. Create DB tables (init_db)
#   2. Sync market data (Yahoo → PostgreSQL)
#   3. Train XGBoost model
#   4. Create or promote drift baseline
#
# Environment variables:
#   SKIP_SYNC=0        (default) Sync data before training
#   SKIP_SYNC=1        Skip sync — use existing DB data (retrain)
#   SKIP_DATA_SYNC=1   Same as SKIP_SYNC=1 (Python native name)
#   CREATE_BASELINE=1  Create baseline for first time
#   PROMOTE_BASELINE=1 Promote baseline (every retrain)
#
# Usage (first time — syncs data + creates baseline):
#   docker compose --profile training run --rm training
#
# Usage (retrain — skip sync, use existing DB data):
#   docker compose --profile training run --rm \
#     -e SKIP_SYNC=1 training
#
# Usage (first time baseline):
#   docker compose --profile training run --rm \
#     -e CREATE_BASELINE=1 training
# ==========================================================

set -e

echo "========================================"
echo " MarketSentinel Training Entrypoint v1.1"
echo "========================================"

# ── FIX: Sync SKIP_SYNC → SKIP_DATA_SYNC ─────────────────
# entrypoint.sh uses SKIP_SYNC (short, docker-friendly name)
# Python code reads SKIP_DATA_SYNC (full descriptive name)
# Export SKIP_DATA_SYNC so Python picks it up correctly.
# If SKIP_DATA_SYNC already set directly, it takes precedence.
if [ "${SKIP_SYNC:-0}" = "1" ] && [ -z "${SKIP_DATA_SYNC}" ]; then
    export SKIP_DATA_SYNC=1
    echo "[entrypoint] SKIP_SYNC=1 → SKIP_DATA_SYNC=1 exported"
fi

# ── Resolve baseline flag ─────────────────────────────────
# If CREATE_BASELINE=1 → first time, create new baseline
# If PROMOTE_BASELINE=1 → retrain, update existing baseline
# Default → promote (safe default for retrain)

BASELINE_FLAG="--promote-baseline"

if [ "${CREATE_BASELINE:-0}" = "1" ]; then
    BASELINE_FLAG="--create-baseline"
    echo "[entrypoint] Mode: CREATE baseline (first time)"
elif [ "${PROMOTE_BASELINE:-1}" = "1" ]; then
    BASELINE_FLAG="--promote-baseline"
    echo "[entrypoint] Mode: PROMOTE baseline (retrain)"
fi

# ── Sync status log ───────────────────────────────────────
if [ "${SKIP_DATA_SYNC:-0}" = "1" ] || [ "${SKIP_SYNC:-0}" = "1" ]; then
    echo "[entrypoint] Data sync: SKIPPED"
else
    echo "[entrypoint] Data sync: ENABLED (set SKIP_SYNC=1 to skip)"
fi

echo "----------------------------------------"
echo "[entrypoint] Starting training pipeline..."
echo "----------------------------------------"

# ── Run training pipeline ─────────────────────────────────
# train_pipeline.py handles: init_db → migration → sync → train → baseline
exec python -m training.pipelines.train_pipeline "$BASELINE_FLAG"