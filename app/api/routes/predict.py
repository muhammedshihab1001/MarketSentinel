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

# ✅ Load once
pipeline = InferencePipeline()

# ------------------------------------------------
# CONCURRENCY GATE
# ------------------------------------------------

MAX_CONCURRENT_INFERENCES = int(
    os.getenv("MAX_CONCURRENT_INFERENCES", "4")
)

inference_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_INFERENCES
)

# 🔥 NEW — portfolio safety
MAX_BATCH_SIZE = int(
    os.getenv("MAX_BATCH_SIZE", "10")
)

# ----------------------------------------
# REQUEST SCHEMAS
# ----------------------------------------

class PredictionRequest(BaseModel):

    ticker: str = Field(
        default="AAPL",
        min_length=1,
        max_length=10
    )

    forecast_days: int = Field(
        default=30,
        ge=1,
        le=90
    )

    start_date: datetime.date | None = None
    end_date: datetime.date | None = None

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


# 🔥 NEW — Batch Schema
class BatchPredictionRequest(BaseModel):

    tickers: list[str] = Field(
        ...,
        min_length=1,
        max_length=50
    )

    forecast_days: int = Field(
        default=30,
        ge=1,
        le=90
    )

    @field_validator("tickers")
    @classmethod
    def normalize(cls, tickers):
        return [t.upper().strip() for t in tickers]


# ----------------------------------------
# SINGLE INFERENCE
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


# ----------------------------------------
# 🔥 PORTFOLIO / BATCH INFERENCE
# ----------------------------------------

@router.post("/predict/batch")
async def predict_batch(req: BatchPredictionRequest):

    endpoint = "/predict/batch"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = time.time()

    if len(req.tickers) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit ({MAX_BATCH_SIZE})"
        )

    async def infer_one(ticker):

        async with inference_semaphore:

            try:

                return await run_in_threadpool(
                    pipeline.run,
                    ticker,
                    None,
                    None,
                    req.forecast_days
                )

            except Exception as e:

                # 🔥 Failure isolation
                return {
                    "ticker": ticker,
                    "error": str(e)
                }

    try:

        results = await asyncio.gather(
            *[infer_one(t) for t in req.tickers]
        )

        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )

        return {
            "count": len(results),
            "results": results
        }

    except Exception as e:

        API_ERROR_COUNT.labels(endpoint=endpoint).inc()

        raise HTTPException(
            status_code=500,
            detail=f"Batch inference failed: {str(e)}"
        )
