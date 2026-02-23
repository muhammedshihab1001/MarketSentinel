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

        snapshot = pipeline.run_snapshot(universe)

        if not isinstance(snapshot, dict) or "signals" not in snapshot:
            raise RuntimeError("Invalid snapshot structure.")

        results = snapshot["signals"]

        if not results:
            raise RuntimeError("No signals generated.")

        # ===============================
        # SIGNAL COUNTS
        # ===============================

        long_count = sum(1 for r in results if r.get("signal") == "LONG")
        short_count = sum(1 for r in results if r.get("signal") == "SHORT")
        neutral_count = sum(1 for r in results if r.get("signal") == "NEUTRAL")

        # ===============================
        # EXPOSURE
        # ===============================

        gross_exposure = sum(abs(r.get("weight", 0.0)) for r in results)
        net_exposure = sum(r.get("weight", 0.0) for r in results)

        # ===============================
        # AGENT METRICS (FIXED)
        # ===============================

        high_conviction_count = sum(
            1 for r in results
            if r.get("agent", {}).get("strength_score", 0.0) >= 75
        )

        elevated_risk_count = sum(
            1 for r in results
            if r.get("agent", {}).get("risk_level") == "elevated"
        )

        return {
            "snapshot_date": snapshot.get("snapshot_date"),
            "universe_size": snapshot.get("universe_size"),
            "long_count": long_count,
            "short_count": short_count,
            "neutral_count": neutral_count,
            "gross_exposure": round(gross_exposure, 6),
            "net_exposure": round(net_exposure, 6),
            "high_conviction_count": high_conviction_count,
            "elevated_risk_count": elevated_risk_count,
        }

    except Exception as e:
        logger.exception("Portfolio summary failed")
        raise HTTPException(status_code=500, detail=str(e))