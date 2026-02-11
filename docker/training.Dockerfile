# -------------------------------------------------
# STAGE 1 — BUILDER
# -------------------------------------------------

FROM python:3.10-slim AS builder

WORKDIR /install

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps ONLY here
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements

RUN pip install --upgrade pip setuptools wheel

# Install into system path inside builder
RUN pip install \
    --no-cache-dir \
    -r requirements/training.txt


# -------------------------------------------------
# STAGE 2 — RUNTIME
# -------------------------------------------------

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime libs ONLY
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy project
COPY core ./core
COPY models ./models
COPY training ./training
COPY requirements ./requirements

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python", "-m", "training.pipelines.train_pipeline"]
