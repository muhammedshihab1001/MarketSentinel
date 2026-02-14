import json
import os
import datetime
import hashlib
import pandas as pd
import platform
import numpy as np

from core.schema.feature_schema import (
    get_schema_signature,
    SCHEMA_VERSION,
    MODEL_FEATURES
)

from core.market.universe import MarketUniverse


class MetadataManager:

    METADATA_VERSION = "11.2"
    MIN_TRAINING_DAYS = 120

    REQUIRED_METADATA_FIELDS = [
        "metadata_type",
        "metadata_version",
        "model_name",
        "created_at",
        "training_window",
        "dataset_hash",
        "dataset_rows",
        "features",
        "feature_count",
        "metrics",
        "schema_signature",
        "schema_version",
        "training_code_hash",
        "metadata_integrity_hash",
        "environment",
        "training_universe",
        "universe_hash"
    ]

    IMMUTABLE_KEYS = {
        "dataset_hash",
        "schema_signature",
        "schema_version",
        "training_code_hash",
        "training_universe",
        "universe_hash",
        "features",
        "feature_count",
    }

    #####################################################

    @staticmethod
    def _fsync_dir_safe(directory: str):

        if os.name == "nt":
            return

        fd = os.open(directory, os.O_DIRECTORY)

        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    #####################################################

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)
        MetadataManager._fsync_dir_safe(directory)

    #####################################################

    @staticmethod
    def load_metadata(path: str) -> dict:

        if os.path.islink(path):
            raise RuntimeError("Symlinked metadata detected.")

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata missing: {path}")

        with open(path) as f:
            metadata = json.load(f)

        missing = [
            k for k in MetadataManager.REQUIRED_METADATA_FIELDS
            if k not in metadata
        ]

        if missing:
            raise RuntimeError(
                f"Metadata missing required fields: {missing}"
            )

        expected = metadata.get("metadata_integrity_hash")
        actual = MetadataManager._compute_metadata_hash(metadata)

        if expected != actual:
            raise RuntimeError(
                "Metadata integrity failure — possible tampering."
            )

        if metadata["metadata_version"] not in ("11.0", "11.1", "11.2"):
            raise RuntimeError("Metadata version mismatch.")

        if metadata["schema_signature"] != get_schema_signature():
            raise RuntimeError("Schema mismatch with runtime.")

        return metadata

    #####################################################

    @staticmethod
    def hash_list(items):

        if not items:
            raise RuntimeError("Cannot hash empty list.")

        normalized = sorted(str(x) for x in items)

        canonical = json.dumps(
            normalized,
            separators=(",", ":")
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    #####################################################

    @staticmethod
    def _validate_feature_contract(features, metadata_type):

        frozen = tuple(features)

        if metadata_type == "training_manifest_v1":

            if frozen != MODEL_FEATURES:
                raise RuntimeError(
                    "Tabular feature mismatch — schema drift."
                )

        elif metadata_type == "timeseries_manifest_v1":

            if list(frozen) != ["close"]:
                raise RuntimeError(
                    "Timeseries models must declare ['close']."
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
    # METRIC VALIDATION
    #####################################################

    @staticmethod
    def _validate_metrics(metrics: dict):

        if not isinstance(metrics, dict) or not metrics:
            raise RuntimeError("Metrics must be a non-empty dict.")

        for k, v in metrics.items():

            if isinstance(v, (list, tuple, dict)):
                raise RuntimeError(
                    f"Metric must be scalar: {k}"
                )

            try:
                val = float(v)
            except Exception:
                raise RuntimeError(
                    f"Metric must be numeric: {k}"
                )

            if not np.isfinite(val):
                raise RuntimeError(
                    f"Non-finite metric detected: {k}"
                )

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

        for col in df_copy.columns:

            if col == "date":
                df_copy[col] = pd.to_datetime(df_copy[col], utc=True)
                continue

            if df_copy[col].dtype == "object":
                df_copy[col] = df_copy[col].astype(str)
                continue

            df_copy[col] = (
                pd.to_numeric(df_copy[col], errors="raise")
                .astype("float64")
                .round(10)
            )

        if not np.isfinite(
            df_copy.select_dtypes(include=[np.number]).to_numpy()
        ).all():
            raise RuntimeError("Non-finite values detected in dataset.")

        hashed = pd.util.hash_pandas_object(
            df_copy,
            index=True
        ).values

        return hashlib.sha256(hashed.tobytes()).hexdigest()

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

                    if os.path.islink(full_path):
                        raise RuntimeError(
                            f"Symlink detected in training code: {full_path}"
                        )

                    rel_path = os.path.relpath(full_path)
                    hasher.update(rel_path.encode())

                    with open(full_path, "rb") as fh:
                        hasher.update(fh.read())

        hasher.update(get_schema_signature().encode())
        hasher.update(platform.python_version().encode())

        return hasher.hexdigest()

    #####################################################

    @staticmethod
    def capture_environment():

        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }

        try:
            import xgboost
            env["xgboost"] = xgboost.__version__
        except Exception:
            pass

        try:
            import torch
            env["torch"] = torch.__version__
        except Exception:
            pass

        try:
            import sklearn
            env["sklearn"] = sklearn.__version__
        except Exception:
            pass

        return env

    #####################################################
    # FLOAT-STABLE HASH
    #####################################################

    @staticmethod
    def _normalize_for_hash(obj):

        if isinstance(obj, float):
            return round(obj, 10)

        if isinstance(obj, dict):
            return {
                k: MetadataManager._normalize_for_hash(v)
                for k, v in sorted(obj.items())
            }

        if isinstance(obj, list):
            return [
                MetadataManager._normalize_for_hash(v)
                for v in obj
            ]

        return obj

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
            raise RuntimeError("Invalid training window.")

        if (end_dt - start_dt).days < MetadataManager.MIN_TRAINING_DAYS:
            raise RuntimeError(
                "Training window below institutional minimum."
            )

    #####################################################

    @staticmethod
    def create_metadata(
        model_name,
        metrics,
        features,
        training_start,
        training_end,
        dataset_hash,
        dataset_rows=None,
        metadata_type=None,
        extra_fields=None
    ):

        if metadata_type is None:
            raise RuntimeError("metadata_type is required.")

        MetadataManager._validate_feature_contract(
            features,
            metadata_type
        )

        MetadataManager._validate_metrics(metrics)

        MetadataManager._validate_training_window(
            training_start,
            training_end
        )

        if dataset_rows is None:
            raise RuntimeError(
                "dataset_rows must be supplied by training pipeline."
            )

        dataset_rows = int(dataset_rows)

        if dataset_rows <= 0:
            raise RuntimeError("dataset_rows must be > 0.")

        universe_snapshot = MarketUniverse.snapshot()

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
            "dataset_rows": dataset_rows,

            "features": list(features),
            "feature_count": len(features),

            "metrics": metrics,

            "schema_signature": get_schema_signature(),
            "schema_version": SCHEMA_VERSION,

            "training_code_hash":
                MetadataManager.fingerprint_training_code(),

            "environment":
                MetadataManager.capture_environment(),

            "training_universe": universe_snapshot,
            "universe_hash": universe_snapshot["universe_hash"]
        }

        if extra_fields:

            forbidden = MetadataManager.IMMUTABLE_KEYS & set(extra_fields)

            if forbidden:
                raise RuntimeError(
                    f"extra_fields cannot override immutable lineage keys: {forbidden}"
                )

            metadata.update(extra_fields)

        metadata["metadata_integrity_hash"] = (
            MetadataManager._compute_metadata_hash(metadata)
        )

        return metadata
