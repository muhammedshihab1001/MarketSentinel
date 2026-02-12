import json
import os
import datetime
import hashlib
import pandas as pd
import platform
import sys
import numpy as np

from core.schema.feature_schema import (
    get_schema_signature,
    SCHEMA_VERSION,
    MODEL_FEATURES
)


class MetadataManager:

    REQUIRED_METADATA_FIELDS = [
        "metadata_type",
        "metadata_version",
        "model_name",
        "created_at",
        "training_window",
        "dataset_hash",
        "features",
        "metrics",
        "training_code_hash"
    ]

    TABULAR_TYPES = {
        "tabular",
        "classification",
        "regression"
    }

    METADATA_VERSION = "2.2"

    #####################################################
    # DATASET FINGERPRINT
    #####################################################

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:

        if df is None or df.empty:
            raise RuntimeError("Cannot fingerprint empty dataset.")

        df_copy = df.copy()

        for col in df_copy.columns:

            if pd.api.types.is_float_dtype(df_copy[col]):
                df_copy[col] = (
                    df_copy[col]
                    .astype("float64")
                    .round(10)
                )

            elif pd.api.types.is_integer_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype("int64")

        sort_cols = []

        if "date" in df_copy.columns:
            sort_cols.append("date")

        if "ticker" in df_copy.columns:
            sort_cols.append("ticker")

        if sort_cols:
            df_copy = df_copy.sort_values(sort_cols)

        df_copy = df_copy.reset_index(drop=True)

        arr = np.ascontiguousarray(
            df_copy.to_numpy()
        )

        return hashlib.sha256(arr.tobytes()).hexdigest()

    #####################################################
    # TRAINING CODE HASH
    #####################################################

    @staticmethod
    def fingerprint_training_code():

        hasher = hashlib.sha256()

        CRITICAL_DIRS = [
            "training",
            "models",
            "core/features",
            "core/schema"
        ]

        for root in CRITICAL_DIRS:

            if not os.path.exists(root):
                continue

            for path, _, files in os.walk(root):

                for f in sorted(files):

                    if f.endswith(".py"):

                        with open(
                            os.path.join(path, f),
                            "rb"
                        ) as fh:
                            hasher.update(fh.read())

        hasher.update(get_schema_signature().encode())

        return hasher.hexdigest()

    #####################################################

    @staticmethod
    def capture_environment():

        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }

        try:
            import sklearn
            env["sklearn"] = sklearn.__version__
        except:
            pass

        try:
            import tensorflow as tf
            env["tensorflow"] = tf.__version__
        except:
            pass

        try:
            import xgboost
            env["xgboost"] = xgboost.__version__
        except:
            pass

        return env

    #####################################################

    @staticmethod
    def create_metadata(
        model_name: str,
        metrics: dict,
        features: list,
        training_start: str,
        training_end: str,
        dataset_hash: str,
        metadata_type: str = "tabular"
    ) -> dict:

        if not isinstance(metrics, dict):
            raise RuntimeError("metrics must be a dictionary.")

        if not isinstance(dataset_hash, str):
            raise RuntimeError("dataset_hash must be a string.")

        if not features:
            raise RuntimeError("features list cannot be empty.")

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

            "training_code_hash":
                MetadataManager.fingerprint_training_code(),

            "environment":
                MetadataManager.capture_environment()
        }

        if metadata_type in MetadataManager.TABULAR_TYPES:

            metadata["schema_signature"] = get_schema_signature()
            metadata["schema_version"] = SCHEMA_VERSION

        MetadataManager.validate_metadata(metadata)

        return metadata

    #####################################################

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

        metadata_type = metadata.get("metadata_type", "").lower()

        if metadata_type in MetadataManager.TABULAR_TYPES:

            if metadata.get("schema_signature") != get_schema_signature():
                raise RuntimeError(
                    "Schema signature mismatch detected."
                )

            if metadata.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(
                    "Schema version mismatch detected."
                )

            if metadata.get("features") != list(MODEL_FEATURES):
                raise RuntimeError(
                    "Tabular feature ordering mismatch."
                )

    #####################################################

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

        fd = os.open(directory or ".", os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)

    #####################################################

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        MetadataManager.validate_metadata(metadata)

        MetadataManager._atomic_json_write(
            path,
            metadata
        )

    #####################################################

    @staticmethod
    def load_metadata(path: str) -> dict:

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata not found: {path}")

        with open(path, "r") as f:
            metadata = json.load(f)

        MetadataManager.validate_metadata(metadata)

        return metadata
