from pydantic import BaseModel, Field, validator
from datetime import date, timedelta


MAX_FORECAST_DAYS = 90


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

    # --------------------------------------------------
    # AUTO VALIDATION (VERY IMPORTANT)
    # --------------------------------------------------

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
