import logging
from fastapi import APIRouter

from core.monitoring.drift_detector import DriftDetector
from app.inference.model_loader import ModelLoader

router = APIRouter()
logger = logging.getLogger("marketsentinel.drift")


@router.get("/drift-status")
def drift_status():

    loader = ModelLoader()
    detector = DriftDetector()

    # Lightweight call – no prediction
    status = detector.get_status()

    return {
        "drift_detected": status.get("drift_detected", False),
        "severity_score": status.get("severity_score", 0.0),
        "model_version": loader.xgb_version
    }