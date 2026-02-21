############################################################
# MarketSentinel — Training Container (Institutional)
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

############################################################
# Build Dependencies (Isolated)
############################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

############################################################
# Copy Only Required Requirements
############################################################

COPY requirements/base.txt requirements/base.txt
COPY requirements/training.txt requirements/training.txt

RUN pip install --upgrade pip setuptools wheel

############################################################
# Install Python Packages
############################################################

RUN pip install --prefer-binary -r requirements/base.txt && \
    pip install --prefer-binary -r requirements/training.txt



############################################################
# STAGE 2 — RUNTIME
############################################################

FROM python:3.10-slim

WORKDIR /app

############################################################
# Runtime Environment (Deterministic)
############################################################

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    APP_ENV=training

############################################################
# Runtime Dependencies Only
############################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tini \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

############################################################
# Non-Root User
############################################################

RUN useradd -m -u 10001 appuser

############################################################
# Copy Installed Python Packages
############################################################

COPY --from=builder /usr/local /usr/local

############################################################
# Copy Project Code (Training Only What’s Needed)
############################################################

COPY core ./core
COPY models ./models
COPY training ./training
COPY config ./config

############################################################
# HuggingFace Cache Directory
############################################################

RUN mkdir -p /app/artifacts/huggingface && \
    chown -R appuser:appuser /app

ENV HF_HOME=/app/artifacts/huggingface

############################################################
# Switch User
############################################################

USER appuser

############################################################
# Tini (Zombie Process Protection)
############################################################

ENTRYPOINT ["/usr/bin/tini", "--"]

############################################################
# Default Command (Training Pipeline)
############################################################

CMD ["python", "-m", "training.train_xgboost"]