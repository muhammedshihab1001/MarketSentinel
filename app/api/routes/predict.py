from fastapi import APIRouter

from app.inference.pipeline import InferencePipeline

router = APIRouter()

pipeline = InferencePipeline()


@router.get("/predict")
def predict(ticker: str = "AAPL"):

    result = pipeline.run(ticker)

    return result
