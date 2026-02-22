import logging
from fastapi import APIRouter

from app.inference.pipeline import InferencePipeline
from core.market.universe import MarketUniverse

router = APIRouter()
logger = logging.getLogger("marketsentinel.portfolio")


@router.get("/portfolio-summary")
def portfolio_summary():

    pipeline = InferencePipeline()
    universe = MarketUniverse.get_universe()

    results = pipeline.run_batch(universe)

    long_count = sum(1 for r in results if r["signal"] == "LONG")
    short_count = sum(1 for r in results if r["signal"] == "SHORT")
    neutral_count = sum(1 for r in results if r["signal"] == "NEUTRAL")

    gross_exposure = sum(abs(r["weight"]) for r in results)
    net_exposure = sum(r["weight"] for r in results)

    return {
        "long_count": long_count,
        "short_count": short_count,
        "neutral_count": neutral_count,
        "gross_exposure": round(gross_exposure, 6),
        "net_exposure": round(net_exposure, 6)
    }