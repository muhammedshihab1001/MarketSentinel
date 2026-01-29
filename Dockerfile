# 1️⃣ Base image
FROM python:3.12-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ System dependencies (for numpy, xgboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Copy requirements
COPY requirements.txt .

# 5️⃣ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy application code
COPY app ./app
COPY training ./training
COPY models ./models

# 7️⃣ Expose FastAPI port
EXPOSE 8000

# 8️⃣ Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
