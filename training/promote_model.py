import os
import json
import logging
import hashlib
from datetime import datetime
import argparse

from core.artifacts.metadata_manager import MetadataManager
from core.monitoring.drift_detector import DriftDetector

logger = logging.getLogger("marketsentinel.promote")

REGISTRY_DIR = os.getenv(
    "XGB_REGISTRY_DIR",
    os.path.abspath("artifacts/xgboost")
)

POINTER_FILENAME = "production_pointer.json"


############################################################
# VALIDATE VERSION
############################################################

def _validate_version_exists(version: str):

    model_path = os.path.join(REGISTRY_DIR, f"model_{version}.pkl")
    metadata_path = os.path.join(REGISTRY_DIR, f"metadata_{version}.json")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Metadata file not found: {metadata_path}")

    # Validate metadata integrity strictly
    metadata = MetadataManager.load_metadata(metadata_path)

    # Optional artifact hash check
    if metadata.get("artifact_hash"):
        actual_hash = _sha256(model_path)
        if metadata["artifact_hash"] != actual_hash:
            raise RuntimeError("Artifact integrity mismatch.")

    return model_path, metadata_path, metadata


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


############################################################
# CHECK CURRENT PRODUCTION
############################################################

def _get_current_production_version():

    pointer_path = os.path.join(REGISTRY_DIR, POINTER_FILENAME)

    if not os.path.exists(pointer_path):
        return None

    with open(pointer_path, encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("model_version")


############################################################
# PROMOTE
############################################################

def promote(version: str, create_baseline=False, force=False):

    if not version:
        raise RuntimeError("Model version required for promotion.")

    if not os.path.exists(REGISTRY_DIR):
        raise RuntimeError("Registry directory missing.")

    current = _get_current_production_version()

    if current == version and not force:
        raise RuntimeError(
            f"Model version {version} already in production."
        )

    model_path, metadata_path, metadata = \
        _validate_version_exists(version)

    pointer_path = os.path.join(REGISTRY_DIR, POINTER_FILENAME)
    tmp_path = pointer_path + ".tmp"

    payload = {
        "model_version": version,
        "promoted_at": datetime.utcnow().isoformat(),
        "dataset_hash": metadata["dataset_hash"],
        "training_code_hash": metadata["training_code_hash"]
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, pointer_path)

    logger.info("Model version %s promoted to production.", version)

    ########################################################
    # OPTIONAL BASELINE CREATION
    ########################################################

    if create_baseline:

        drift = DriftDetector()

        # Load training dataset metadata for reference only
        logger.info("Creating baseline for promoted model.")

        # NOTE: baseline must be manually created via training
        # We do not recreate full dataset here.
        # This ensures governance separation.

        raise RuntimeError(
            "Baseline creation must be done during training "
            "with --create-baseline flag."
        )


############################################################
# CLI
############################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=str)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    promote(
        version=args.version,
        force=args.force
    )