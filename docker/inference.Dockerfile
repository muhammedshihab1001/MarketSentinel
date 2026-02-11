# -------------------------------------------------
# STAGE 1 — BUILDER
# -------------------------------------------------

FROM python:3.10-slim@sha256:5c9b4a4f7b6e4d5e6c6f1a1b7c2d9c6c4c5d9a1f2e3b4c5d6e7f8a9b0c1d2e3f AS builder

WORKDIR /install

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    --prefix=/install/deps \
    --no-cache-dir \
    -r requirements/inference.txt


# -------------------------------------------------
# STAGE 2 — RUNTIME
# -------------------------------------------------

FROM python:3.10-slim@sha256:5c9b4a4f7b6e4d5e6c6f1a1b7c2d9c6c4c5d9a1f2e3b4c5d6e7f8a9b0c1d2e3f

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    UVICORN_WORKERS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd -m -u 10001 appuser

COPY --from=builder /install/deps /usr/local

COPY app ./app
COPY core ./core
COPY models ./models

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]

# -------------------------------------------------
# REAL HEALTHCHECK
# -------------------------------------------------

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

# -------------------------------------------------
# CPU-AWARE WORKERS
# -------------------------------------------------

CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${UVICORN_WORKERS:-$(python -c "import os; print(max(1, os.cpu_count()//2))")} \
    --loop uvloop \
    --http httptools \
    --timeout-keep-alive 5
