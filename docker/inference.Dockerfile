############################################################
# MarketSentinel — Inference Container
# v2.0: Multi-stage build (matches training.Dockerfile)
#
# STAGE 1 (builder): installs all build tools + pip packages
# STAGE 2 (runtime): copies only compiled packages, no gcc/g++
#
# Estimated savings vs single-stage: 200-350 MB
#
# HOW TO USE:
#   Standard build (no LLM):
#     docker build -f docker/inference.Dockerfile -t marketsentinel-api .
#
#   With LLM support:
#     docker build --build-arg INSTALL_LLM=true \
#       -f docker/inference.Dockerfile -t marketsentinel-api .
############################################################


############################################################
# STAGE 1 — BUILDER
# Installs build tools + compiles all Python packages
# This stage is thrown away — only /usr/local is copied out
############################################################

FROM python:3.10-slim AS builder

WORKDIR /install

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    XGBOOST_NO_CUDA=1

# Build tools needed to compile psycopg2, cryptography etc.
# These stay in the builder stage only — NOT in final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/

# FIX: Install xgboost[cpu] first to lock the CPU-only wheel
# before the full requirements install resolves dependencies.
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary "xgboost[cpu]==2.1.1" && \
    pip install --prefer-binary -r requirements/inference.txt

# Optional LLM support — only installed if INSTALL_LLM=true
ARG INSTALL_LLM=false
RUN if [ "$INSTALL_LLM" = "true" ]; then \
    pip install -r requirements/llm.txt; fi


############################################################
# STAGE 2 — RUNTIME
# Clean slim image — only compiled packages copied from builder
# No gcc, no g++, no build-essential in final image
############################################################

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    XGBOOST_NO_CUDA=1 \
    UVICORN_WORKERS=1 \
    APP_ENV=production

# Only runtime libs — no compiler toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 10001 appuser

# Copy ONLY compiled Python packages from builder — not the build tools
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY app ./app
COPY core ./core
COPY config ./config
COPY requirements ./requirements

# Create required directories and set ownership
RUN mkdir -p /app/artifacts /app/logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# tini handles zombie processes (same as training.Dockerfile)
ENTRYPOINT ["/usr/bin/tini", "--"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--loop", "uvloop", \
    "--http", "httptools", \
    "--timeout-keep-alive", "30"]