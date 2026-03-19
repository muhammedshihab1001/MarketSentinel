
# ==========================================================
# MarketSentinel — Training Container Entrypoint v1.0
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
#   CREATE_BASELINE=1  Create baseline for first time
#   PROMOTE_BASELINE=1 Promote baseline (every retrain)
#
# Usage (first time):
#   docker-compose run --rm training
#
# Usage (retrain, data already in DB):
#   SKIP_SYNC=1 docker-compose run --rm training
#
# Usage (override command entirely):
#   docker-compose run --rm training python -c "..."
# ==========================================================

set -e

echo "========================================"
echo " MarketSentinel Training Entrypoint v1.0"
echo "========================================"

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

# ── Sync flag ─────────────────────────────────────────────
if [ "${SKIP_SYNC:-0}" = "1" ]; then
    echo "[entrypoint] Data sync: SKIPPED (SKIP_SYNC=1)"
else
    echo "[entrypoint] Data sync: ENABLED (set SKIP_SYNC=1 to skip on retrain)"
fi

echo "----------------------------------------"
echo "[entrypoint] Starting training pipeline..."
echo "----------------------------------------"

# ── Run training pipeline ─────────────────────────────────
# train_pipeline.py handles: init_db → migration → sync → train → baseline
exec python -m training.pipelines.train_pipeline "$BASELINE_FLAG"