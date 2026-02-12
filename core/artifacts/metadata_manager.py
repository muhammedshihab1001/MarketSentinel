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
    SCHEMA_VERSION
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
        "schema_signature",
        "schema_version",
        "training_code_hash"
    ]

    METADATA_VERSION = "2.2"

    ########################################################
    # SAFE DIRECTORY FSYNC
    ########################################################

    @staticmethod
    def _fsync_dir_safe(directory: str):

        if os.name == "nt":
            return

        try:
            fd = os.open(directory, os.O_DIRECTORY)
            os.fsync(fd)
            os.close(fd)
        except Exception:
            pass

    ########################################################
    # DATASET FINGERPRINT (STRUCTURALLY DETERMINISTIC)
    ########################################################

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:

        if df is None or df.empty:
            raise RuntimeError("Cannot fingerprint empty dataset.")

        df_copy = df.copy()

        # enforce deterministic column ordering
        df_copy = df_copy.reindex(sorted(df_copy.columns), axis=1)

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

        hasher = hashlib.sha256()

        # include column structure
        hasher.update(",".join(df_copy.columns).encode())

        arr = np.ascontiguousarray(
            df_copy.to_numpy()
        )

        hasher.update(arr.tobytes())

        return hasher.hexdigest()

    ########################################################
    # TRAINING CODE HASH (PATH + CONTENT)
    ########################################################

    @staticmethod
    def fingerprint_training_code():

        hasher = hashlib.sha256()

        CRITICAL_DIRS = [
            "training",
            "models",
            "core/features",
            "core/schema"
        ]

        for root in sorted(CRITICAL_DIRS):

            if not os.path.exists(root):
                continue

            for path, dirs, files in os.walk(root):

                dirs.sort()
                files.sort()

                for f in files:

                    if not f.endswith(".py"):
                        continue

                    full_path = os.path.join(path, f)
                    rel_path = os.path.relpath(full_path)

                    hasher.update(rel_path.encode())

                    with open(full_path, "rb") as fh:
                        hasher.update(fh.read())

        hasher.update(get_schema_signature().encode())

        return hasher.hexdigest()

    ########################################################

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
        except Exception:
            pass

        try:
            import tensorflow as tf
            env["tensorflow"] = tf.__version__
        except Exception:
            pass

        try:
            import xgboost
            env["xgboost"] = xgboost.__version__
        except Exception:
            pass

        return env

    ########################################################

    @staticmethod
    def create_metadata(
        model_name: str,
        metrics: dict,
        features: list,
        training_start: str,
        training_end: str,
        dataset_hash: str,
        metadata_type: str = "model",
        extra_fields: dict | None = None
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

            "training_code_hash":
                MetadataManager.fingerprint_training_code(),

            "environment":
                MetadataManager.capture_environment()
        }

        if extra_fields:
            metadata.update(extra_fields)

        MetadataManager.validate_metadata(metadata)

        return metadata

    ########################################################

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
            raise RuntimeError("Metadata features must be a list.")

        if not isinstance(metadata["metrics"], dict):
            raise RuntimeError("Metadata metrics must be a dict.")

        if not isinstance(metadata["dataset_hash"], str):
            raise RuntimeError("Dataset hash must be a string.")

        if metadata["schema_signature"] != get_schema_signature():
            raise RuntimeError(
                "Schema signature mismatch detected."
            )

        if metadata["schema_version"] != SCHEMA_VERSION:
            raise RuntimeError(
                "Schema version mismatch detected."
            )

    ########################################################
    # ATOMIC WRITE
    ########################################################

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

        MetadataManager._fsync_dir_safe(directory or ".")

    ########################################################

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        MetadataManager.validate_metadata(metadata)

        MetadataManager._atomic_json_write(
            path,
            metadata
        )

    ########################################################

    @staticmethod
    def load_metadata(path: str) -> dict:

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata not found: {path}")

        with open(path, "r") as f:
            metadata = json.load(f)

        MetadataManager.validate_metadata(metadata)

        return metadata
