import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.inference.model_loader import ModelLoader
from core.schema.feature_schema import MODEL_FEATURES

router = APIRouter()
logger = logging.getLogger("marketsentinel.model_info")


# =========================================================
# MODEL INFO
# =========================================================

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


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

@router.get("/feature-importance")
def feature_importance():

    loader = ModelLoader()

    try:
        result = loader.get_feature_importance()

        return {
            "model_version": result["model_version"],
            "feature_checksum": result["feature_checksum"],
            "importance_checksum": result["importance"]["importance_checksum"],
            "feature_importance": result["importance"]["feature_importance"]
        }

    except Exception as e:
        logger.exception("Feature importance retrieval failed")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# MODEL DIAGNOSTICS (CV SHOWCASE)
# =========================================================

@router.get("/model-diagnostics")
def model_diagnostics() -> Dict[str, Any]:

    loader = ModelLoader()

    try:
        model = loader.xgb

        diagnostics = {
            "model_version": loader.xgb_version,
            "artifact_hash": loader.artifact_hash,
            "schema_signature": loader.schema_signature,
            "dataset_hash": loader.dataset_hash,
            "training_code_hash": loader.training_code_hash,
            "feature_checksum": loader.feature_checksum,
            "feature_count": len(MODEL_FEATURES),
            "training_rows": getattr(model, "training_rows", None),
            "training_cols": getattr(model, "training_cols", None),
            "param_checksum": getattr(model, "param_checksum", None),
            "training_checksum": getattr(model, "training_checksum", None),
        }

        return diagnostics

    except Exception as e:
        logger.exception("Model diagnostics failure")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )