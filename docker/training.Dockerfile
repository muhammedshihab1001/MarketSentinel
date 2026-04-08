############################################################
# MarketSentinel — Training Container v2.2
#
# Changes from v2.1:
#   - Uses docker/entrypoint.sh for automated boot sequence
#   - entrypoint.sh handles: init_db → sync → train → baseline
#   - Single docker-compose run --rm training does everything
#   - SKIP_SYNC=1 env var skips sync on retrain
#   - CREATE_BASELINE=1 for first run, PROMOTE_BASELINE=1 default
############################################################

############################################################
# STAGE 1 — BUILDER
############################################################

FROM python:3.10-slim AS builder

WORKDIR /install

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/

# Install xgboost[cpu] first to prevent nvidia-nccl-cu12 (294MB) download
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary "xgboost[cpu]==2.1.1" && \
    pip install --prefer-binary -r requirements/training.txt


############################################################
# STAGE 2 — RUNTIME
############################################################

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    APP_ENV=training \
    PYTHONPATH=/app \
    LANG=C.UTF-8 \
    # Default: promote baseline (retrain mode)
    # Override with CREATE_BASELINE=1 for first run
    PROMOTE_BASELINE=1 \
    # Default: sync data before training
    # Override with SKIP_SYNC=1 to skip on retrain
    SKIP_SYNC=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libpq5 tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 appuser

COPY --from=builder /usr/local /usr/local

COPY core ./core
COPY training ./training
COPY config ./config
COPY requirements ./requirements
COPY docker/entrypoint.sh ./entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh && \
    mkdir -p /app/artifacts /app/data /app/logs && \
    chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]