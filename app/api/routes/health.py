import time
import logging
from fastapi import APIRouter

from app.monitoring.metrics import (
    API_REQUEST_COUNT,
    API_LATENCY,
    API_ERROR_COUNT
)

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger("marketsentinel.health")


@router.get("/live")
def liveness():
    """
    Basic container liveness probe.
    Used by orchestrators to verify service is alive.
    """

    endpoint = "/health/live"
    API_REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        return {
            "status": "alive",
            "service": "MarketSentinel"
        }

    except Exception:
        API_ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.exception("Health check failure")
        return {
            "status": "error"
        }

    finally:
        API_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - start_time
        )