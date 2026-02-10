# -------------------------------------------------
# STAGE 1 — Builder
# -------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /install

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build tools ONLY here
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements

RUN pip install --upgrade pip

# Install dependencies into a temp directory
RUN pip install \
    --prefix=/install \
    --no-cache-dir \
    -r requirements/training.txt


# -------------------------------------------------
# STAGE 2 — Runtime Image (LEAN)
# -------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy ONLY installed packages
COPY --from=builder /install /usr/local

# Copy ONLY required folders
COPY core ./core
COPY models ./models
COPY training ./training
COPY requirements ./requirements

CMD ["python", "-m", "training.pipelines.train_pipeline"]
