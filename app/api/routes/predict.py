from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import time

from app.inference.pipeline import InferencePipeline
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()

pipeline = InferencePipeline()


# ----------------------------------------
# REQUEST SCHEMA
# ----------------------------------------

class PredictionRequest(BaseModel):

    ticker: str = Field(
        default="AAPL",
        min_length=1,
        max_length=10
    )


# ----------------------------------------
# INFERENCE ROUTE
# ----------------------------------------

@router.post("/predict")
def predict(req: PredictionRequest):

    endpoint = "/predict"

    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start = time.time()

    try:

        result = pipeline.run(req.ticker.upper())

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
