############################################################
# MarketSentinel — Inference Container (Production)
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
    APP_ENV=production

WORKDIR /app

############################################################
# System Dependencies (Minimal Runtime Only)
############################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

############################################################
# Python Dependencies
############################################################

COPY requirements/base.txt requirements/base.txt
COPY requirements/inference.txt requirements/inference.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements/base.txt && \
    pip install --no-cache-dir -r requirements/inference.txt

############################################################
# Non-Root User
############################################################

RUN useradd -m -u 10001 appuser

############################################################
# Application Code (Runtime Only)
############################################################

COPY app ./app
COPY core ./core
COPY models ./models
COPY config ./config

# Artifacts directory created but NOT populated
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
CMD curl -f http://127.0.0.1:8000/live || exit 1

############################################################
# Server
############################################################

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools"]