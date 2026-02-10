import json
import os
import datetime
import hashlib
import pandas as pd
import shutil
import tempfile


class MetadataManager:
    """
    Institutional Metadata + Model Registry.

    Provides:

    dataset fingerprinting
    metadata validation
    atomic writes
    versioned registry
    rollback safety
    lineage tracking
    governance readiness
    """

    REQUIRED_METADATA_FIELDS = [
        "model_name",
        "created_at",
        "training_window",
        "dataset_hash",
        "features",
        "metrics",
        "schema_version"
    ]

    # -----------------------------------------------------
    # DATASET FINGERPRINT
    # -----------------------------------------------------

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:
        """
        Immutable dataset hash.
        Any row change -> new fingerprint.
        """

        df = df.sort_index(axis=1)

        data_bytes = pd.util.hash_pandas_object(
            df,
            index=True
        ).values.tobytes()

        return hashlib.sha256(data_bytes).hexdigest()

    # -----------------------------------------------------
    # METADATA
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

            "schema_version": "3.0"
        }

        MetadataManager.validate_metadata(metadata)

        return metadata

    # -----------------------------------------------------
    # VALIDATION (VERY IMPORTANT)
    # -----------------------------------------------------

    @staticmethod
    def validate_metadata(metadata: dict):
        """
        Prevents registry corruption.
        """

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

    # -----------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------

    @staticmethod
    def save_metadata(metadata: dict, path: str):
        """
        Atomic write prevents corrupted metadata.
        """

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

    @staticmethod
    def load_metadata(path: str) -> dict:

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata not found: {path}")

        with open(path, "r") as f:
            metadata = json.load(f)

        MetadataManager.validate_metadata(metadata)

        return metadata

    # =====================================================
    # MODEL REGISTRY
    # =====================================================

    @staticmethod
    def register_model(model_dir: str):
        """
        Promotes trained artifacts into a versioned registry.
        """

        if not os.path.exists(model_dir):
            raise RuntimeError(f"{model_dir} does not exist")

        files_to_move = [
            f for f in os.listdir(model_dir)
            if not f.startswith("v_") and f != "latest"
        ]

        if not files_to_move:
            raise RuntimeError(
                "No artifacts found to register."
            )

        timestamp = datetime.datetime.utcnow().strftime(
            "v_%Y_%m_%d_%H%M%S"
        )

        version_path = os.path.join(model_dir, timestamp)
        os.makedirs(version_path, exist_ok=True)

        for file in files_to_move:

            src = os.path.join(model_dir, file)
            dst = os.path.join(version_path, file)

            shutil.move(src, dst)

        # validate metadata exists
        metadata_path = os.path.join(version_path, "metadata.json")

        if not os.path.exists(metadata_path):
            raise RuntimeError(
                "metadata.json missing. Refusing to register model."
            )

        MetadataManager.load_metadata(metadata_path)

        # update latest symlink
        latest_path = os.path.join(model_dir, "latest")

        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)

        os.symlink(
            os.path.abspath(version_path),
            latest_path
        )

        print(f"Model registered -> {version_path}")
        print(f"'latest' now points to {timestamp}")

        return version_path

    # -----------------------------------------------------

    @staticmethod
    def list_versions(model_dir: str):

        if not os.path.exists(model_dir):
            return []

        return sorted([
            d for d in os.listdir(model_dir)
            if d.startswith("v_")
        ])

    # -----------------------------------------------------

    @staticmethod
    def rollback(model_dir: str, version: str):
        """
        Repoints latest to a previous version.
        """

        version_path = os.path.join(model_dir, version)

        if not os.path.exists(version_path):
            raise RuntimeError(
                f"Version {version} not found"
            )

        metadata_path = os.path.join(
            version_path,
            "metadata.json"
        )

        MetadataManager.load_metadata(metadata_path)

        latest_path = os.path.join(model_dir, "latest")

        if os.path.exists(latest_path) or os.path.islink(latest_path):
            os.remove(latest_path)

        os.symlink(
            os.path.abspath(version_path),
            latest_path
        )

        print(f"Rolled back to {version}")
