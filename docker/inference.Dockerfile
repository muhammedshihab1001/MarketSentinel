# -------------------------------------------------
# STAGE 1 — Builder
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
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements

# deterministic dependency resolution
RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    --prefix=/install/deps \
    -r requirements/inference.txt


# -------------------------------------------------
# STAGE 2 — Runtime
# -------------------------------------------------
FROM python:3.10-slim@sha256:5c9b4a4f7b6e4d5e6c6f1a1b7c2d9c6c4c5d9a1f2e3b4c5d6e7f8a9b0c1d2e3f

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# runtime-only deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd -m appuser

COPY --from=builder /install/deps /usr/local

COPY app ./app
COPY core ./core
COPY models ./models

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# container-level health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('127.0.0.1',8000)); s.close()"

# production uvicorn config
CMD ["uvicorn", "app.main:app","--host", "0.0.0.0","--port", "8000","--workers", "2","--loop", "uvloop","--http", "httptools","--timeout-keep-alive", "5"]
