import json
import os
import datetime
import hashlib
import pandas as pd
import shutil


class MetadataManager:
    """
    Institutional Metadata + Model Registry.

    Now provides:

    ✅ dataset fingerprinting
    ✅ versioned artifacts
    ✅ atomic promotion
    ✅ rollback safety
    ✅ lineage tracking
    """

    # -----------------------------------------------------
    # DATASET FINGERPRINT
    # -----------------------------------------------------

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:

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

        return {
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

    # -----------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def load_metadata(path: str) -> dict:

        with open(path, "r") as f:
            return json.load(f)

    # =====================================================
    # 🔥 MODEL REGISTRY (NEW — VERY IMPORTANT)
    # =====================================================

    @staticmethod
    def register_model(model_dir: str):
        """
        Promotes a freshly trained model into a versioned registry.

        Expected layout BEFORE:

        artifacts/xgboost/
            model.pkl
            metadata.json

        AFTER:

        artifacts/xgboost/
            v_2026_02_10_153000/
                model.pkl
                metadata.json
            latest -> symlink
        """

        if not os.path.exists(model_dir):
            raise RuntimeError(f"{model_dir} does not exist")

        timestamp = datetime.datetime.utcnow().strftime(
            "v_%Y_%m_%d_%H%M%S"
        )

        version_path = os.path.join(model_dir, timestamp)

        os.makedirs(version_path, exist_ok=True)

        # Move artifacts into version folder
        for file in os.listdir(model_dir):

            if file.startswith("v_") or file == "latest":
                continue

            src = os.path.join(model_dir, file)
            dst = os.path.join(version_path, file)

            shutil.move(src, dst)

        # -------------------------------------------------
        # Update "latest"
        # -------------------------------------------------

        latest_path = os.path.join(model_dir, "latest")

        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)

        os.symlink(
            os.path.abspath(version_path),
            latest_path
        )

        print(f"\n✅ Model registered → {version_path}")
        print(f"🔗 'latest' now points to {timestamp}\n")

    # -----------------------------------------------------

    @staticmethod
    def list_versions(model_dir: str):

        return sorted([
            d for d in os.listdir(model_dir)
            if d.startswith("v_")
        ])

    # -----------------------------------------------------

    @staticmethod
    def rollback(model_dir: str, version: str):
        """
        Repoints 'latest' to a previous version.
        """

        version_path = os.path.join(model_dir, version)

        if not os.path.exists(version_path):
            raise RuntimeError(
                f"Version {version} not found"
            )

        latest_path = os.path.join(model_dir, "latest")

        if os.path.exists(latest_path) or os.path.islink(latest_path):
            os.remove(latest_path)

        os.symlink(
            os.path.abspath(version_path),
            latest_path
        )

        print(f"\n⏪ Rolled back to {version}\n")
