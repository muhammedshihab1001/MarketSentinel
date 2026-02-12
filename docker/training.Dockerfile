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
# BUILD DEPENDENCIES
############################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements

RUN pip install --upgrade pip setuptools wheel

############################################################
# INSTALL PYTHON PACKAGES
############################################################

RUN pip install \
    --prefer-binary \
    -r requirements/training.txt


############################################################
# STAGE 2 — RUNTIME
############################################################

FROM python:3.10-slim

WORKDIR /app

############################################################
# RUNTIME ENV (VERY IMPORTANT)
############################################################

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

############################################################
# RUNTIME LIBS
############################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tini \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

############################################################
# NON ROOT USER (SECURITY)
############################################################

RUN useradd -m appuser

############################################################
# COPY PACKAGES FROM BUILDER
############################################################

COPY --from=builder /usr/local /usr/local

############################################################
# COPY PROJECT
############################################################

COPY core ./core
COPY models ./models
COPY training ./training
COPY requirements ./requirements

############################################################
# HUGGINGFACE CACHE DIRECTORY
############################################################

RUN mkdir -p /app/artifacts/huggingface \
    && chown -R appuser:appuser /app

ENV HF_HOME=/app/artifacts/huggingface

############################################################
# SWITCH USER
############################################################

USER appuser

############################################################
# TINI (ZOMBIE PROCESS FIX)
############################################################

ENTRYPOINT ["/usr/bin/tini", "--"]

############################################################
# TRAIN PIPELINE
############################################################

CMD ["python", "-m", "training.pipelines.train_pipeline"]
