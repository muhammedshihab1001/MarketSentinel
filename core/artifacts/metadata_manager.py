import json
import os
import datetime


class MetadataManager:
    """
    Handles model metadata creation.

    Lightweight alternative to MLflow.
    Enables traceability and governance.
    """

    @staticmethod
    def create_metadata(
        model_name: str,
        metrics: dict,
        features: list,
        training_start: str,
        training_end: str
    ) -> dict:

        metadata = {
            "model_name": model_name,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "training_window": {
                "start": training_start,
                "end": training_end
            },
            "features": features,
            "metrics": metrics,
            "schema_version": "1.0"
        }

        return metadata

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)
