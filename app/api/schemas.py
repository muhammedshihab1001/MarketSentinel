from pydantic import BaseModel, Field, field_validator
from datetime import date
from typing import List, Optional, Dict, Any


MAX_FORECAST_DAYS = 90


# =========================================================
# FORECAST REQUEST (LEGACY / OPTIONAL)
# =========================================================

class PredictionRequest(BaseModel):
    """
    Optional request schema used by older endpoints.
    Kept for backward compatibility.
    """

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

    # Primary model agent score
    agent_score: float

    # Hybrid consensus score
    hybrid_score: Optional[float] = None

    # Technical confirmation score
    technical_score: Optional[float] = None

    alpha_strength: float
    confidence_numeric: float
    governance_score: int

    risk_level: str
    volatility_regime: str

    drift_flag: bool

    warnings: List[str] = Field(default_factory=list)
    explanation: str


# =========================================================
# META INFORMATION
# =========================================================

class SignalExplanationMeta(BaseModel):

    model_config = {
        "protected_namespaces": ()
    }

    model_version: Optional[str] = None
    schema_signature: Optional[str] = None
    dataset_hash: Optional[str] = None
    artifact_hash: Optional[str] = None

    latency_ms: int
    timestamp: int


# =========================================================
# SIGNAL EXPLANATION ENVELOPE
# =========================================================

class SignalExplanationEnvelope(BaseModel):

    meta: SignalExplanationMeta
    explanation: SignalExplanationResponse


# =========================================================
# LIVE SNAPSHOT RESPONSE
# =========================================================

class LiveSnapshotResponse(BaseModel):

    meta: Dict[str, Any]
    executive_summary: Dict[str, Any]
    snapshot: Dict[str, Any]


# =========================================================
# AGENT EXPLAIN RESPONSE
# =========================================================

class AgentExplainResponse(BaseModel):

    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str


# =========================================================
# DRIFT STATUS RESPONSE
# =========================================================

class DriftStatusResponse(BaseModel):

    drift_state: str
    severity_score: float
    timestamp: int


# =========================================================
# PORTFOLIO RESPONSE
# =========================================================

class PortfolioResponse(BaseModel):

    top_positions: List[Dict[str, Any]]
    gross_exposure: float
    net_exposure: float
    timestamp: int