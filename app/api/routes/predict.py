from fastapi import APIRouter, HTTPException
import time

from app.inference.pipeline import InferencePipeline
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()
pipeline = InferencePipeline()


@router.get("/predict")
def predict(ticker: str = "AAPL"):

    endpoint = "/predict"

    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start = time.time()

    try:

        result = pipeline.run(ticker)

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start
        )

        return result

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
