import json
import os
import datetime
import hashlib
import pandas as pd


class MetadataManager:
    """
    Institutional metadata manager.

    Upgrades:
    ✅ Dataset fingerprinting
    ✅ Artifact versioning
    ✅ Reproducibility
    ✅ Governance-ready
    """

    # -----------------------------------------------------

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:
        """
        Creates immutable dataset hash.

        If ANY row changes → hash changes.

        Critical for:
        - reproducibility
        - audits
        - MLflow migration
        """

        # stable serialization
        data_bytes = pd.util.hash_pandas_object(
            df,
            index=True
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

            "schema_version": "2.0"
        }

        return metadata

    # -----------------------------------------------------

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------------------------------

    @staticmethod
    def load_metadata(path: str) -> dict:

        with open(path, "r") as f:
            return json.load(f)
