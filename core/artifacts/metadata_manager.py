import json
import os
import datetime
import hashlib
import pandas as pd
import platform
import sys


from core.schema.feature_schema import (
    get_schema_signature,
    SCHEMA_VERSION
)


class MetadataManager:
    """
    Institutional Metadata Authority.

    Guarantees:
    - deterministic dataset fingerprinting
    - crash-safe writes
    - schema-bound metadata
    - forward migration safety
    - registry poisoning prevention
    - audit-grade environment capture
    """

    REQUIRED_METADATA_FIELDS = [
        "metadata_type",
        "metadata_version",
        "model_name",
        "created_at",
        "training_window",
        "dataset_hash",
        "features",
        "metrics",
        "schema_signature",
        "schema_version"
    ]

    METADATA_VERSION = "1.1"

    # -----------------------------------------------------
    # DATASET FINGERPRINT (DETERMINISTIC)
    # -----------------------------------------------------

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:
        """
        Produces a stable dataset fingerprint across:
        - machines
        - CPU architectures
        - pandas versions (within reason)
        """

        if df is None or df.empty:
            raise RuntimeError("Cannot fingerprint empty dataset.")

        df_copy = df.copy()

        # stable column ordering
        df_copy = df_copy.reindex(sorted(df_copy.columns), axis=1)

        # normalize floats to avoid CPU rounding drift
        float_cols = df_copy.select_dtypes(include=["float32", "float64"]).columns
        df_copy[float_cols] = df_copy[float_cols].round(10)

        # deterministic row ordering
        df_copy = df_copy.sort_values(
            by=list(df_copy.columns)
        ).reset_index(drop=True)

        hashed = pd.util.hash_pandas_object(
            df_copy,
            index=False
        ).values.tobytes()

        return hashlib.sha256(hashed).hexdigest()

    # -----------------------------------------------------

    @staticmethod
    def create_metadata(
        model_name: str,
        metrics: dict,
        features: list,
        training_start: str,
        training_end: str,
        dataset_hash: str,
        metadata_type: str = "model"
    ) -> dict:

        metadata = {

            "metadata_type": metadata_type,
            "metadata_version": MetadataManager.METADATA_VERSION,

            "model_name": model_name,
            "created_at": datetime.datetime.utcnow().isoformat(),

            "training_window": {
                "start": training_start,
                "end": training_end
            },

            "dataset_hash": dataset_hash,
            "features": features,
            "metrics": metrics,

            "schema_signature": get_schema_signature(),
            "schema_version": SCHEMA_VERSION,

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

        if metadata["metadata_type"] != "model":
            raise RuntimeError(
                "Only model metadata allowed for registry."
            )

        if metadata["schema_signature"] != get_schema_signature():
            raise RuntimeError(
                "Schema signature mismatch detected."
            )

        if metadata["schema_version"] != SCHEMA_VERSION:
            raise RuntimeError(
                "Schema version mismatch detected."
            )

        if not isinstance(metadata["features"], list):
            raise RuntimeError("features must be a list")

        if not isinstance(metadata["metrics"], dict):
            raise RuntimeError("metrics must be a dictionary")

        if not isinstance(metadata["dataset_hash"], str):
            raise RuntimeError("dataset_hash must be a string")

    # -----------------------------------------------------
    # CRASH-SAFE WRITE
    # -----------------------------------------------------

    @staticmethod
    def _atomic_json_write(path: str, payload: dict):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    # -----------------------------------------------------

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        MetadataManager.validate_metadata(metadata)

        MetadataManager._atomic_json_write(
            path,
            metadata
        )

    # -----------------------------------------------------

    @staticmethod
    def load_metadata(path: str) -> dict:

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata not found: {path}")

        with open(path, "r") as f:
            metadata = json.load(f)

        MetadataManager.validate_metadata(metadata)

        return metadata
