# -------------------------------------------------
# Base Image
# -------------------------------------------------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# -------------------------------------------------
# System Dependencies (minimal)
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Install ONLY inference dependencies
# -------------------------------------------------
COPY requirements ./requirements

RUN pip install --no-cache-dir -r requirements/inference.txt

# -------------------------------------------------
# Copy runtime code ONLY
# -------------------------------------------------
COPY app ./app
COPY core ./core
COPY models ./models
COPY artifacts ./artifacts

# -------------------------------------------------
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
