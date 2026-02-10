from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.concurrency import run_in_threadpool
import datetime
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

    forecast_days: int = Field(
        default=30,
        ge=1,
        le=90
    )

    start_date: datetime.date | None = None
    end_date: datetime.date | None = None


    # ✅ Pydantic v2 style
    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str):
        return v.upper()


    # ✅ Date validation (v2)
    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v, info):

        start = info.data.get("start_date")

        if start and v:
            if v <= start:
                raise ValueError("end_date must be after start_date")

        return v


# ----------------------------------------
# ASYNC INFERENCE ROUTE
# ----------------------------------------

@router.post("/predict")
async def predict(req: PredictionRequest):

    endpoint = "/predict"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start = time.time()

    try:

        # ⭐ RUN BLOCKING ML SAFELY
        result = await run_in_threadpool(
            pipeline.run,
            req.ticker,
            req.start_date,
            req.end_date,
            req.forecast_days
        )

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
