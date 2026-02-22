import logging
from fastapi import APIRouter

from app.inference.model_loader import ModelLoader
from core.schema.feature_schema import MODEL_FEATURES

router = APIRouter()
logger = logging.getLogger("marketsentinel.model_info")


@router.get("/model-info")
def model_info():

    loader = ModelLoader()

    return {
        "model_version": loader.xgb_version,
        "schema_signature": loader.schema_signature,
        "dataset_hash": loader.dataset_hash,
        "training_code_hash": loader.training_code_hash,
        "artifact_hash": loader.artifact_hash,
        "feature_checksum": loader.feature_checksum,
        "feature_count": len(MODEL_FEATURES)
    }