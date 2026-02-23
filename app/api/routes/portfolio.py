import logging
from fastapi import APIRouter, HTTPException

from app.inference.pipeline import InferencePipeline
from core.market.universe import MarketUniverse

router = APIRouter()
logger = logging.getLogger("marketsentinel.portfolio")


@router.get("/portfolio-summary")
def portfolio_summary():

    try:
        pipeline = InferencePipeline()
        universe = MarketUniverse.get_universe()

        # Use canonical snapshot path
        snapshot = pipeline.run_snapshot(universe)
        results = snapshot.get("signals", [])

        if not results:
            raise RuntimeError("No signals generated.")

        long_count = sum(1 for r in results if r["signal"] == "LONG")
        short_count = sum(1 for r in results if r["signal"] == "SHORT")
        neutral_count = sum(1 for r in results if r["signal"] == "NEUTRAL")

        gross_exposure = sum(abs(r.get("weight", 0.0)) for r in results)
        net_exposure = sum(r.get("weight", 0.0) for r in results)

        return {
            "snapshot_date": snapshot.get("snapshot_date"),
            "universe_size": snapshot.get("universe_size"),
            "long_count": long_count,
            "short_count": short_count,
            "neutral_count": neutral_count,
            "gross_exposure": round(gross_exposure, 6),
            "net_exposure": round(net_exposure, 6),
            "high_conviction_count": snapshot.get("high_conviction_count"),
            "elevated_risk_count": snapshot.get("elevated_risk_count"),
        }

    except Exception as e:
        logger.exception("Portfolio summary failed")
        raise HTTPException(status_code=500, detail=str(e))