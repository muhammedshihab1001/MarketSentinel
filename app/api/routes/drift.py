import logging
from fastapi import APIRouter

from app.inference.pipeline import InferencePipeline
from core.market.universe import MarketUniverse
from core.schema.feature_schema import MODEL_FEATURES

router = APIRouter()
logger = logging.getLogger("marketsentinel.drift")


@router.get("/drift-status")
def drift_status():

    pipeline = InferencePipeline()

    try:
        tickers = MarketUniverse.get_universe()

        df = pipeline._build_cross_sectional_frame(tickers)
        latest_df = pipeline._select_latest_snapshot(df)

        feature_df = latest_df.loc[:, MODEL_FEATURES]

        drift_result = pipeline.drift_detector.detect(feature_df)

        return {
            "drift_detected": drift_result.get("drift_detected", False),
            "severity_score": drift_result.get("severity_score", 0),
            "model_version": pipeline.models.xgb_version
        }

    except Exception:
        logger.exception("Drift status check failed")

        return {
            "drift_detected": True,
            "severity_score": 10,
            "reason": "status_failure"
        }