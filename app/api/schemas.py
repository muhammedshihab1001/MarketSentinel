from pydantic import BaseModel, Field, field_validator
from datetime import date
from typing import List, Optional


MAX_FORECAST_DAYS = 90


# =========================================================
# FORECAST REQUEST
# =========================================================

class PredictionRequest(BaseModel):

    ticker: str = Field(
        default="AAPL",
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g. AAPL, TSLA)"
    )

    start_date: Optional[date] = Field(
        default=None,
        description="Forecast start date (default = today)"
    )

    end_date: Optional[date] = Field(
        default=None,
        description="Forecast end date"
    )

    forecast_days: Optional[int] = Field(
        default=30,
        ge=1,
        le=MAX_FORECAST_DAYS,
        description="Number of days to forecast (max 90)"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.upper()

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v, info):
        start = info.data.get("start_date")

        if v and start and v <= start:
            raise ValueError("end_date must be after start_date")

        if v and start:
            delta = (v - start).days
            if delta > MAX_FORECAST_DAYS:
                raise ValueError(
                    f"Forecast window cannot exceed {MAX_FORECAST_DAYS} days"
                )

        return v

    @field_validator("start_date", mode="before")
    @classmethod
    def default_start(cls, v):
        return v or date.today()

    @field_validator("forecast_days")
    @classmethod
    def validate_forecast_days(cls, v):
        if v and v > MAX_FORECAST_DAYS:
            raise ValueError(
                f"forecast_days cannot exceed {MAX_FORECAST_DAYS}"
            )
        return v


# =========================================================
# MULTI-AGENT SIGNAL EXPLANATION
# =========================================================

class SignalExplanationResponse(BaseModel):

    ticker: str
    score: float
    signal: str

    # Multi-agent core metrics
    agent_score: float
    alpha_strength: float
    confidence_numeric: float
    governance_score: int

    risk_level: str
    volatility_regime: str

    drift_flag: bool

    warnings: List[str] = []
    explanation: str


# =========================================================
# META INFORMATION
# =========================================================

class SignalExplanationMeta(BaseModel):

    model_config = {
        "protected_namespaces": ()
    }

    model_version: str
    schema_signature: str
    dataset_hash: str
    artifact_hash: str
    latency_ms: int
    timestamp: int


# =========================================================
# ENVELOPE
# =========================================================

class SignalExplanationEnvelope(BaseModel):

    meta: SignalExplanationMeta
    explanation: SignalExplanationResponse