############################################################
# MarketSentinel — Inference Container (CV-Optimized)
############################################################

FROM python:3.10-slim

############################################################
# Environment
############################################################

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    UVICORN_WORKERS=1 \
    APP_ENV=production

WORKDIR /app

############################################################
# System Dependencies
############################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tini \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

############################################################
# Python Dependencies (Layer Optimized)
############################################################

COPY requirements/ ./requirements/

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary -r requirements/inference.txt

############################################################
# Remove build dependencies
############################################################

RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

############################################################
# Non-Root User
############################################################

RUN useradd -m -u 10001 appuser

############################################################
# Application Code
############################################################

COPY app ./app
COPY core ./core
COPY config ./config

RUN mkdir -p /app/artifacts && \
    chown -R appuser:appuser /app

USER appuser

############################################################
# Networking
############################################################

EXPOSE 8000

############################################################
# Init Process
############################################################

ENTRYPOINT ["/usr/bin/tini", "--"]

############################################################
# Healthcheck
############################################################

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
CMD curl -f http://127.0.0.1:8000/health/ready || exit 1

############################################################
# Server
############################################################

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1","--loop","uvloop","--http","httptools","--timeout-keep-alive","30"]