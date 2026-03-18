############################################################
# MarketSentinel — Training Container (CV-Optimized)
# FIX: Added XGBOOST_NO_CUDA=1 — removes 293MB GPU dep
############################################################

############################################################
# STAGE 1 — BUILDER
############################################################

FROM python:3.10-slim AS builder

WORKDIR /install

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    XGBOOST_NO_CUDA=1

############################################################
# Build Dependencies
############################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

############################################################
# Copy Requirements
############################################################

COPY requirements/ ./requirements/

############################################################
# Install Python Packages
############################################################

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary -r requirements/training.txt


############################################################
# STAGE 2 — RUNTIME
############################################################

FROM python:3.10-slim

WORKDIR /app

############################################################
# Runtime Environment
# XGBOOST_NO_CUDA=1 prevents GPU package download at runtime
############################################################

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
    XGBOOST_NO_CUDA=1

############################################################
# Runtime System Packages
############################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    tini \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

############################################################
# Non-Root User
############################################################

RUN useradd -m -u 10001 appuser

############################################################
# Copy Installed Python Packages from Builder
############################################################

COPY --from=builder /usr/local /usr/local

############################################################
# Copy Project Code
############################################################

COPY core ./core
COPY training ./training
COPY config ./config
COPY requirements ./requirements

############################################################
# Artifacts + Data + Logs Directories
############################################################

RUN mkdir -p /app/artifacts /app/data /app/logs && \
    chown -R appuser:appuser /app

############################################################
# Switch User
############################################################

USER appuser

############################################################
# Init (Zombie Protection)
############################################################

ENTRYPOINT ["/usr/bin/tini", "--"]

############################################################
# Default Training Command
############################################################

CMD ["python", "-m", "training.pipelines.train_pipeline"]