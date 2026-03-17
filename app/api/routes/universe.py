# =========================================================
# UNIVERSE ROUTE v2.1
# DB-Aware | CV-Optimized
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

router = APIRouter()


# =========================================================
# UNIVERSE INFO ENDPOINT
# =========================================================

@router.get("/universe")
def universe_info():
    """
    Returns the current stock universe: tickers, version, hash.
    """

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