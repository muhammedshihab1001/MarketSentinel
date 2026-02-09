# -------------------------------------------------
# 1️⃣ Base image (slim = smaller, faster)
# -------------------------------------------------
FROM python:3.12-slim

# -------------------------------------------------
# 2️⃣ Environment settings (best practice)
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------------------------------
# 3️⃣ Set working directory
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# 4️⃣ System dependencies
# Required for:
# - numpy
# - pandas
# - tensorflow
# - prophet
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 5️⃣ Copy minimal runtime requirements
# (IMPORTANT: keeps Docker image small)
# -------------------------------------------------
COPY requirements/inference.txt .

# -------------------------------------------------
# 6️⃣ Install Python dependencies
# --no-cache-dir reduces image size
# -------------------------------------------------
RUN pip install --no-cache-dir -r inference.txt


# -------------------------------------------------
# 7️⃣ Copy application source code
# Only runtime-required folders
# -------------------------------------------------
COPY app ./app
COPY models ./models

# -------------------------------------------------
# 8️⃣ Expose FastAPI port
# -------------------------------------------------
EXPOSE 8000

# -------------------------------------------------
# 9️⃣ Start FastAPI application
# -------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
