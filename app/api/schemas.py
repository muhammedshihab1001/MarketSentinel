from pydantic import BaseModel, Field, validator
from datetime import date
from typing import List, Optional, Dict, Any


MAX_FORECAST_DAYS = 90


# =========================================================
# EXISTING FORECAST REQUEST (UNCHANGED)
# =========================================================

class PredictionRequest(BaseModel):
    """
    Request schema for stock forecasting.
    Designed for production-safe inference.
    """

    ticker: str = Field(
        default="AAPL",
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g. AAPL, TSLA)"
    )

    start_date: date | None = Field(
        default=None,
        description="Forecast start date (default = today)"
    )

    end_date: date | None = Field(
        default=None,
        description="Forecast end date"
    )

    forecast_days: int | None = Field(
        default=30,
        ge=1,
        le=MAX_FORECAST_DAYS,
        description="Number of days to forecast (max 90)"
    )

    @validator("ticker")
    def normalize_ticker(cls, v):
        return v.upper()

    @validator("end_date")
    def validate_dates(cls, v, values):

        start = values.get("start_date")

        if v and start and v <= start:
            raise ValueError("end_date must be after start_date")

        if v and start:
            delta = (v - start).days

            if delta > MAX_FORECAST_DAYS:
                raise ValueError(
                    f"Forecast window cannot exceed {MAX_FORECAST_DAYS} days"
                )

        return v

    @validator("start_date", pre=True, always=True)
    def default_start(cls, v):
        return v or date.today()

    @validator("forecast_days")
    def validate_forecast_days(cls, v):
        if v and v > MAX_FORECAST_DAYS:
            raise ValueError(
                f"forecast_days cannot exceed {MAX_FORECAST_DAYS}"
            )
        return v


# =========================================================
# NEW SIGNAL EXPLANATION RESPONSE MODEL
# =========================================================

class SignalExplanationResponse(BaseModel):
    """
    Detailed explanation output for a single ticker signal.
    """

    ticker: str
    score: float
    rank_pct: float
    signal: str
    strength_score: float
    risk_level: str
    confidence: str
    volatility_regime: str
    trend: str
    momentum_state: str
    warnings: List[str]
    explanation: str


class SignalExplanationMeta(BaseModel):
    """
    Metadata block for explanation endpoint.
    """

    model_version: str
    schema_signature: str
    dataset_hash: str
    training_code_hash: str
    artifact_hash: str
    latency_ms: int
    timestamp: int


class SignalExplanationEnvelope(BaseModel):
    """
    Full API response structure.
    """

    meta: SignalExplanationMeta
    explanation: SignalExplanationResponse