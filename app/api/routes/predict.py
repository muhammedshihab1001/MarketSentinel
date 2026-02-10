from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.concurrency import run_in_threadpool
import datetime
import time
import asyncio
import os

from app.inference.pipeline import InferencePipeline
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter()

# ✅ Load once (VERY important)
pipeline = InferencePipeline()

# ------------------------------------------------
# INSTITUTIONAL CONCURRENCY GATE
# ------------------------------------------------
MAX_CONCURRENT_INFERENCES = int(
    os.getenv("MAX_CONCURRENT_INFERENCES", "4")
)

inference_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_INFERENCES
)


# ----------------------------------------
# REQUEST SCHEMA
# ----------------------------------------

class PredictionRequest(BaseModel):

    ticker: str = Field(
        default="AAPL",
        min_length=1,
        max_length=10,
        description="Stock ticker symbol"
    )

    forecast_days: int = Field(
        default=30,
        ge=1,
        le=90,
        description="Number of days to forecast (max 90)"
    )

    start_date: datetime.date | None = Field(
        default=None,
        description="Optional forecast start date"
    )

    end_date: datetime.date | None = Field(
        default=None,
        description="Optional forecast end date"
    )

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str):
        return v.upper().strip()

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v, info):

        start = info.data.get("start_date")

        if start and v and v <= start:
            raise ValueError("end_date must be after start_date")

        return v


# ----------------------------------------
# ASYNC INFERENCE ROUTE
# ----------------------------------------

@router.post("/predict")
async def predict(req: PredictionRequest):

    endpoint = "/predict"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = time.time()

    try:

        async with inference_semaphore:

            result = await run_in_threadpool(
                pipeline.run,
                req.ticker,
                req.start_date,
                req.end_date,
                req.forecast_days
            )

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )

        return result

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
