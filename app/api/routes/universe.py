# =========================================================
# UNIVERSE ROUTE v2.2
# SWAGGER FIX: Added tags, summary, description
# =========================================================

import time
from fastapi import APIRouter, HTTPException

from core.market.universe import MarketUniverse
from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT,
)
from core.logging.logger import get_logger

logger = get_logger("marketsentinel.universe")

router = APIRouter(tags=["universe"])


@router.get(
    "/universe",
    summary="Stock Universe",
    description="""
Returns the current S&P 500 stock universe used by the inference pipeline.

**Response includes:**
- `tickers`: list of all 100 ticker symbols (use these for equity, performance, agent endpoints)
- `count`: number of tickers (should be 100)
- `version`: universe config version
- `universe_hash`: SHA256 hash of the ticker list

**No authentication required.**
""",
    response_description="List of all universe tickers with version and hash.",
)
def universe_info():
    endpoint = "/universe"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        snapshot = MarketUniverse.snapshot()

        if not isinstance(snapshot, dict):
            raise RuntimeError("Invalid universe snapshot structure.")

        tickers = snapshot.get("tickers", [])

        logger.debug(
            "Universe snapshot | tickers=%d | version=%s",
            len(tickers),
            snapshot.get("version"),
        )

        return {
            "version": snapshot.get("version"),
            "description": snapshot.get("description"),
            "tickers": tickers,
            "count": len(tickers),
            "universe_hash": snapshot.get("universe_hash"),
        }

    except Exception as e:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Universe endpoint failure")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)