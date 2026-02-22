import os
import json
import logging
from datetime import datetime

from core.artifacts.metadata_manager import MetadataManager

logger = logging.getLogger("marketsentinel.promote")

REGISTRY_DIR = os.getenv(
    "XGB_REGISTRY_DIR",
    os.path.abspath("artifacts/xgboost")
)

POINTER_FILENAME = "production_pointer.json"


def _validate_version_exists(version: str):
    model_path = os.path.join(REGISTRY_DIR, f"model_{version}.pkl")
    metadata_path = os.path.join(REGISTRY_DIR, f"metadata_{version}.json")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Metadata file not found: {metadata_path}")

    # Validate metadata integrity strictly
    MetadataManager.load_metadata(metadata_path)

    return model_path, metadata_path


def promote(version: str):

    if not version:
        raise RuntimeError("Model version required for promotion.")

    if not os.path.exists(REGISTRY_DIR):
        raise RuntimeError("Registry directory missing.")

    _validate_version_exists(version)

    pointer_path = os.path.join(REGISTRY_DIR, POINTER_FILENAME)
    tmp_path = pointer_path + ".tmp"

    payload = {
        "model_version": version,
        "promoted_at": datetime.utcnow().isoformat()
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, pointer_path)

    logger.info("Model version %s promoted to production.", version)


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python promote_model.py <version>")

    promote(sys.argv[1])