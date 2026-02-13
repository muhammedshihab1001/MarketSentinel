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

from core.market.universe import MarketUniverse


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
        "training_code_hash",
        "metadata_integrity_hash",
        "environment",
        "training_universe",
        "universe_hash"   # ⭐ NEW
    ]

    METADATA_VERSION = "8.0"

    #####################################################
    # SAFE FSYNC
    #####################################################

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

    #####################################################
    # FEATURE CONTRACT
    #####################################################

    @staticmethod
    def _validate_feature_contract(features, metadata_type):

        frozen = tuple(features)

        if metadata_type == "training_manifest_v1":

            if frozen != MODEL_FEATURES:
                raise RuntimeError(
                    "Feature list mismatch — schema drift detected."
                )

        elif metadata_type == "timeseries_manifest_v1":

            if list(frozen) != ["close"]:
                raise RuntimeError(
                    "Timeseries models must declare ['close'] only."
                )

        elif metadata_type == "sequence_manifest_v1":

            if list(frozen) != ["close_sequence"]:
                raise RuntimeError(
                    "Sequence models must declare ['close_sequence']."
                )

        else:
            raise RuntimeError(
                f"Unknown metadata_type: {metadata_type}"
            )

    #####################################################
    # DATASET FINGERPRINT (DETERMINISTIC)
    #####################################################

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:

        if df is None or df.empty:
            raise RuntimeError("Cannot fingerprint empty dataset.")

        df_copy = df.copy(deep=True)

        if "ticker" in df_copy.columns:
            df_copy = df_copy.sort_values(["ticker", "date"])
        elif "date" in df_copy.columns:
            df_copy = df_copy.sort_values("date")

        df_copy = df_copy.reset_index(drop=True)

        #################################################
        # FLOAT NORMALIZATION — CRITICAL
        #################################################

        for col in df_copy.columns:

            if col == "date":
                df_copy[col] = pd.to_datetime(
                    df_copy[col],
                    utc=True
                )
                continue

            df_copy[col] = (
                pd.to_numeric(df_copy[col], errors="raise")
                .astype("float64")
                .round(10)
            )

        hashed = pd.util.hash_pandas_object(
            df_copy,
            index=True
        ).values

        return hashlib.sha256(hashed.tobytes()).hexdigest()

    #####################################################
    # TRAINING CODE HASH (STRICT)
    #####################################################

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
            "core/market",
        ]

        for root in sorted(CRITICAL_DIRS):

            if not os.path.exists(root):
                continue

            for path, dirs, files in os.walk(root):

                dirs.sort()
                files = sorted(f for f in files if f.endswith(".py"))

                for f in files:

                    full_path = os.path.join(path, f)
                    rel_path = os.path.relpath(full_path)

                    hasher.update(rel_path.encode())

                    with open(full_path, "rb") as fh:
                        hasher.update(fh.read())

        hasher.update(get_schema_signature().encode())
        hasher.update(platform.python_version().encode())

        return hasher.hexdigest()

    #####################################################
    # ENVIRONMENT (REPRODUCIBLE)
    #####################################################

    @staticmethod
    def capture_environment():

        env = {
            "python": platform.python_version(),
            "platform": platform.system(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }

        # optional libs

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

    #####################################################
    # NORMALIZATION FOR HASH
    #####################################################

    @staticmethod
    def _normalize_for_hash(obj):

        if isinstance(obj, dict):
            return {k: MetadataManager._normalize_for_hash(v)
                    for k, v in sorted(obj.items())}

        if isinstance(obj, list):
            return [MetadataManager._normalize_for_hash(v) for v in obj]

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            return float(obj)

        return obj

    #####################################################

    @staticmethod
    def _compute_metadata_hash(metadata: dict):

        clone = dict(metadata)
        clone.pop("metadata_integrity_hash", None)

        normalized = MetadataManager._normalize_for_hash(clone)

        canonical = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":")
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    #####################################################

    @staticmethod
    def _validate_training_window(start, end):

        start_dt = pd.to_datetime(start, utc=True)
        end_dt = pd.to_datetime(end, utc=True)

        if end_dt <= start_dt:
            raise RuntimeError(
                "Invalid training window."
            )

    #####################################################

    @staticmethod
    def _validate_universe(universe):

        if not universe:
            raise RuntimeError("Training universe is empty.")

        if universe != sorted(universe):
            raise RuntimeError(
                "Universe snapshot must be sorted."
            )

    #####################################################
    # CREATE METADATA
    #####################################################

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

        MetadataManager._validate_training_window(
            training_start,
            training_end
        )

        universe = MarketUniverse.snapshot()

        MetadataManager._validate_universe(universe)

        universe_hash = hashlib.sha256(
            json.dumps(universe).encode()
        ).hexdigest()

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
            "features": list(tuple(features)),
            "metrics": metrics,

            "schema_signature": get_schema_signature(),
            "schema_version": SCHEMA_VERSION,

            "training_code_hash":
                MetadataManager.fingerprint_training_code(),

            "environment":
                MetadataManager.capture_environment(),

            "training_universe": universe,
            "universe_hash": universe_hash
        }

        if extra_fields:
            metadata.update(extra_fields)

        metadata["metadata_integrity_hash"] = (
            MetadataManager._compute_metadata_hash(metadata)
        )

        MetadataManager.validate_metadata(metadata)

        return metadata

    #####################################################

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

    #####################################################

    @staticmethod
    def _atomic_json_write(path, payload):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        tmp = path + ".tmp"

        try:

            with open(tmp, "w") as f:
                json.dump(payload, f, indent=4, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, path)

            MetadataManager._fsync_dir_safe(directory or ".")

        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    #####################################################

    @staticmethod
    def save_metadata(metadata, path):

        MetadataManager.validate_metadata(metadata)
        MetadataManager._atomic_json_write(path, metadata)

    #####################################################

    @staticmethod
    def load_metadata(path):

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata not found: {path}")

        with open(path, "r") as f:
            metadata = json.load(f)

        MetadataManager.validate_metadata(metadata)

        return metadata
