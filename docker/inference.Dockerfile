# -------------------------------------------------
# STAGE 1 — Builder
# -------------------------------------------------
FROM python:3.10.14 AS builder
# ↑ Get the REAL digest from Docker Hub (shown below)

WORKDIR /install

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements

RUN pip install --upgrade pip

RUN pip install \
    --prefix=/install/deps \
    --no-cache-dir \
    -r requirements/inference.txt


# -------------------------------------------------
# STAGE 2 — Runtime
# -------------------------------------------------
FROM python:3.10.14-slim@sha256:3f1b...REPLACE_ME...

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install/deps /usr/local

COPY app ./app
COPY core ./core
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
