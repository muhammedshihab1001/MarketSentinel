FROM python:3.10

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements

RUN pip install --no-cache-dir -r requirements/training.txt

COPY . .

CMD ["python", "-m", "training.pipelines.train_pipeline"]
