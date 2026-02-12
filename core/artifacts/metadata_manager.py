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

from core.market.universe import MarketUniverse   # ⭐ NEW


class MetadataManager:

    ########################################################
    # REQUIRED FIELDS (INSTITUTIONAL)
    ########################################################

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
        "training_code_hash",
        "metadata_integrity_hash",
        "environment",

        # ⭐ NEW (CRITICAL)
        "training_universe"
    ]

    METADATA_VERSION = "6.0"   # ← bump again


    ########################################################
    # FSYNC SAFE
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
    # FEATURE CONTRACT
    ########################################################

    @staticmethod
    def _validate_feature_contract(features, metadata_type):

        if metadata_type == "training_manifest_v1":

            if tuple(features) != tuple(MODEL_FEATURES):
                raise RuntimeError(
                    "Feature list mismatch — schema drift detected."
                )

        elif metadata_type == "timeseries_manifest_v1":

            if features != ["close"]:
                raise RuntimeError(
                    "Timeseries models must declare ['close'] only."
                )

        elif metadata_type == "sequence_manifest_v1":

            if features != ["close_sequence"]:
                raise RuntimeError(
                    "Sequence models must declare ['close_sequence']."
                )

        else:
            raise RuntimeError(
                f"Unknown metadata_type: {metadata_type}"
            )


    ########################################################
    # DATASET FINGERPRINT
    ########################################################

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:

        if df is None or df.empty:
            raise RuntimeError("Cannot fingerprint empty dataset.")

        df_copy = df.copy(deep=True)
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

            else:
                df_copy[col] = df_copy[col].astype(str)

        sort_cols = []

        if "date" in df_copy.columns:
            sort_cols.append("date")

        if "ticker" in df_copy.columns:
            sort_cols.append("ticker")

        if sort_cols:
            df_copy = df_copy.sort_values(sort_cols)

        df_copy = df_copy.reset_index(drop=True)

        canonical = json.dumps(
            df_copy.to_dict(orient="records"),
            sort_keys=True
        ).encode()

        return hashlib.sha256(canonical).hexdigest()


    ########################################################
    # TRAINING CODE HASH
    ########################################################

    @staticmethod
    def fingerprint_training_code():

        hasher = hashlib.sha256()

        CRITICAL_DIRS = [
            "training",
            "models",
            "core/features",
            "core/schema",
            "core/data",
            "core/time",
            "core/market",   # ⭐ ensures universe changes trigger hash
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
        hasher.update(platform.python_version().encode())

        return hasher.hexdigest()


    ########################################################
    # ENV CAPTURE
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
            import xgboost
            env["xgboost"] = xgboost.__version__
        except Exception:
            pass

        try:
            import tensorflow
            env["tensorflow"] = tensorflow.__version__
        except Exception:
            pass

        return env


    ########################################################
    # METADATA HASH
    ########################################################

    @staticmethod
    def _compute_metadata_hash(metadata: dict):

        clone = dict(metadata)
        clone.pop("metadata_integrity_hash", None)

        canonical = json.dumps(
            clone,
            sort_keys=True
        ).encode()

        return hashlib.sha256(canonical).hexdigest()


    ########################################################
    # CREATE METADATA (INSTITUTIONAL)
    ########################################################

    @staticmethod
    def create_metadata(
        model_name,
        metrics,
        features,
        training_start,
        training_end,
        dataset_hash,
        metadata_type,
        extra_fields=None
    ):

        MetadataManager._validate_feature_contract(
            features,
            metadata_type
        )

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
                MetadataManager.capture_environment(),

            ##################################################
            # ⭐ CRITICAL — UNIVERSE LINEAGE
            ##################################################
            "training_universe":
                MarketUniverse.snapshot()
        }

        if extra_fields:
            metadata.update(extra_fields)

        metadata["metadata_integrity_hash"] = (
            MetadataManager._compute_metadata_hash(metadata)
        )

        MetadataManager.validate_metadata(metadata)

        return metadata


    ########################################################
    # VALIDATE
    ########################################################

    @staticmethod
    def validate_metadata(metadata):

        missing = [
            f for f in MetadataManager.REQUIRED_METADATA_FIELDS
            if f not in metadata
        ]

        if missing:
            raise RuntimeError(
                f"Metadata missing required fields: {missing}"
            )

        if metadata["metadata_integrity_hash"] != (
            MetadataManager._compute_metadata_hash(metadata)
        ):
            raise RuntimeError(
                "Metadata integrity failure detected."
            )

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
    def _atomic_json_write(path, payload):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

        MetadataManager._fsync_dir_safe(directory or ".")


    ########################################################

    @staticmethod
    def save_metadata(metadata, path):

        MetadataManager.validate_metadata(metadata)
        MetadataManager._atomic_json_write(path, metadata)


    ########################################################

    @staticmethod
    def load_metadata(path):

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata not found: {path}")

        with open(path, "r") as f:
            metadata = json.load(f)

        MetadataManager.validate_metadata(metadata)

        return metadata
