# -------------------------------------------------
# STAGE 1 — Builder
# -------------------------------------------------
FROM python:3.10 AS builder

WORKDIR /install

# System deps only for building wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++

COPY requirements ./requirements

# Upgrade pip first (important)
RUN pip install --upgrade pip

# Install dependencies into a custom directory
RUN pip install --prefix=/install/deps --no-cache-dir \
    -r requirements/inference.txt


# -------------------------------------------------
# STAGE 2 — Runtime (SLIM + CLEAN)
# -------------------------------------------------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Only runtime libs (VERY minimal)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Copy ONLY runtime code
COPY app ./app
COPY core ./core
COPY models ./models

# ❗ DO NOT COPY ARTIFACTS
# They will be mounted via docker-compose

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
