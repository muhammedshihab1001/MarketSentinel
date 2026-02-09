from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):

    ticker: str = Field(
        default="AAPL",
        min_length=1,
        max_length=10
    )
