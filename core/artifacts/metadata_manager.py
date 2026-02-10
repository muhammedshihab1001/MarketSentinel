import json
import os
import datetime
import hashlib
import pandas as pd
import tempfile
import shutil
import platform
import sys

from core.schema.feature_schema import get_schema_signature, SCHEMA_VERSION


class MetadataManager:
    """
    Institutional Metadata Manager.

    STRICTLY handles metadata.

    DOES NOT control model promotion.
    """

    REQUIRED_METADATA_FIELDS = [
        "model_name",
        "created_at",
        "training_window",
        "dataset_hash",
        "features",
        "metrics",
        "schema_signature",
        "schema_version"
    ]

    # -----------------------------------------------------
    # DATASET FINGERPRINT
    # -----------------------------------------------------

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:

        df = df.sort_index(axis=1).sort_values(
            by=list(df.columns)
        ).reset_index(drop=True)

        data_bytes = pd.util.hash_pandas_object(
            df,
            index=False
        ).values.tobytes()

        return hashlib.sha256(data_bytes).hexdigest()

    # -----------------------------------------------------

    @staticmethod
    def create_metadata(
        model_name: str,
        metrics: dict,
        features: list,
        training_start: str,
        training_end: str,
        dataset_hash: str
    ) -> dict:

        metadata = {

            "model_name": model_name,
            "created_at": datetime.datetime.utcnow().isoformat(),

            "training_window": {
                "start": training_start,
                "end": training_end
            },

            "dataset_hash": dataset_hash,
            "features": features,
            "metrics": metrics,

            # GOVERNANCE CRITICAL
            "schema_signature": get_schema_signature(),
            "schema_version": SCHEMA_VERSION,

            # FUTURE MIGRATION SAFETY
            "metadata_version": "1.0",

            # AUDIT GOLD
            "environment": {
                "python": sys.version,
                "platform": platform.platform()
            }
        }

        MetadataManager.validate_metadata(metadata)

        return metadata

    # -----------------------------------------------------

    @staticmethod
    def validate_metadata(metadata: dict):

        missing = [
            field for field in MetadataManager.REQUIRED_METADATA_FIELDS
            if field not in metadata
        ]

        if missing:
            raise RuntimeError(
                f"Metadata missing required fields: {missing}"
            )

        if not isinstance(metadata["features"], list):
            raise RuntimeError("features must be a list")

        if not isinstance(metadata["metrics"], dict):
            raise RuntimeError("metrics must be a dictionary")

        if not isinstance(metadata["dataset_hash"], str):
            raise RuntimeError("dataset_hash must be a string")

        if not isinstance(metadata["schema_signature"], str):
            raise RuntimeError("schema_signature must be a string")

    # -----------------------------------------------------

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        MetadataManager.validate_metadata(metadata)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=os.path.dirname(path)
        ) as tmp:

            json.dump(metadata, tmp, indent=4)
            temp_name = tmp.name

        shutil.move(temp_name, path)

    # -----------------------------------------------------

    @staticmethod
    def load_metadata(path: str) -> dict:

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata not found: {path}")

        with open(path, "r") as f:
            metadata = json.load(f)

        MetadataManager.validate_metadata(metadata)

        return metadata
