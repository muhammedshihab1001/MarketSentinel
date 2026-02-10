from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
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

    # 🔥 Normalize ticker
    @validator("ticker")
    def uppercase_ticker(cls, v):
        return v.upper()

    # 🔥 Validate dates
    @validator("end_date")
    def validate_dates(cls, v, values):

        start = values.get("start_date")

        if start and v:
            if v <= start:
                raise ValueError("end_date must be after start_date")

        return v


# ----------------------------------------
# INFERENCE ROUTE
# ----------------------------------------

@router.post("/predict")
def predict(req: PredictionRequest):

    endpoint = "/predict"

    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start = time.time()

    try:

        result = pipeline.run(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            forecast_days=req.forecast_days
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
