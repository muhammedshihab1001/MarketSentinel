from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def liveness():
    """
    Basic container liveness probe.
    Used by orchestrators to verify service is alive.
    """
    return {
        "status": "alive",
        "service": "MarketSentinel"
    }