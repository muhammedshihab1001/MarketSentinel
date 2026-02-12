from fastapi import APIRouter
from app.inference.model_loader import ModelLoader

router = APIRouter()

# Load once
models = None


# -------------------------------------------------
# LIVENESS
# -------------------------------------------------

@router.get("/")
def liveness():
    """
    Simple container check.
    Used by orchestrators to know the service is alive.
    """
    return {
        "status": "alive",
        "service": "MarketSentinel"
    }


# -------------------------------------------------
# READINESS
# -------------------------------------------------

@router.get("/ready")
def readiness():
    """
    Verifies that models and artifacts are loaded.
    Critical for production ML systems.
    """

    global models

    try:

        if models is None:
            models = ModelLoader()

        # Basic checks
        assert models.xgb is not None
        assert models.lstm is not None

        return {
            "status": "ready",
            "inference": "available"
        }

    except Exception as e:

        return {
            "status": "not_ready",
            "error": str(e)
        }
