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

    METADATA_VERSION = "12.2"
    MIN_TRAINING_DAYS = 120
    MIN_METADATA_BYTES = 800
    MIN_FEATURE_COUNT = 10

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
        "environment",
        "training_universe",
        "universe_hash",
        "metadata_integrity_hash"
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
    # ATOMIC SAVE
    #####################################################

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)

        tmp = path + ".tmp"

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    #####################################################

    @staticmethod
    def load_metadata(path: str) -> dict:

        if os.path.islink(path):
            raise RuntimeError("Symlinked metadata detected.")

        if not os.path.exists(path):
            raise RuntimeError(f"Metadata missing: {path}")

        if os.path.getsize(path) < MetadataManager.MIN_METADATA_BYTES:
            raise RuntimeError("Metadata file suspiciously small.")

        with open(path, encoding="utf-8") as f:
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

        if metadata["schema_signature"] != get_schema_signature():
            raise RuntimeError("Schema mismatch with runtime.")

        if metadata["schema_version"] != SCHEMA_VERSION:
            raise RuntimeError("Schema version drift detected.")

        if metadata["features"] != MODEL_FEATURES:
            raise RuntimeError("Metadata feature contract mismatch.")

        if metadata["feature_count"] != len(MODEL_FEATURES):
            raise RuntimeError("Feature count mismatch.")

        if metadata["dataset_rows"] <= 0:
            raise RuntimeError("Invalid dataset_rows in metadata.")

        return metadata

    #####################################################
    # HASH HELPERS
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
    # FEATURE CONTRACT
    #####################################################

    @staticmethod
    def _validate_feature_contract(features, metadata_type):

        frozen = tuple(features)

        if metadata_type != "training_manifest_v1":
            raise RuntimeError(f"Unknown metadata_type: {metadata_type}")

        if frozen != tuple(MODEL_FEATURES):
            raise RuntimeError("Tabular feature mismatch — schema drift.")

        if len(frozen) < MetadataManager.MIN_FEATURE_COUNT:
            raise RuntimeError(
                "Feature contract below institutional minimum."
            )

    #####################################################
    # METRIC VALIDATION
    #####################################################

    @staticmethod
    def _validate_metrics(metrics: dict):

        if not isinstance(metrics, dict) or not metrics:
            raise RuntimeError("Metrics must be a non-empty dict.")

        for k, v in metrics.items():
            try:
                val = float(v)
            except Exception:
                raise RuntimeError(f"Metric must be numeric: {k}")

            if not np.isfinite(val):
                raise RuntimeError(f"Non-finite metric detected: {k}")

    #####################################################
    # DATASET HASH
    #####################################################

    @staticmethod
    def fingerprint_dataset(df: pd.DataFrame) -> str:

        if df is None or df.empty:
            raise RuntimeError("Cannot fingerprint empty dataset.")

        df = df.copy()

        if "ticker" in df.columns:
            df = df.sort_values(["ticker", "date"])
        else:
            df = df.sort_values("date")

        df = df.reset_index(drop=True)

        for col in df.columns:

            if col == "date":
                df[col] = pd.to_datetime(df[col], utc=True)
                continue

            if df[col].dtype == "object":
                df[col] = df[col].astype(str)
                continue

            df[col] = (
                pd.to_numeric(df[col], errors="raise")
                .astype("float64")
                .round(8)
            )

        arr = pd.util.hash_pandas_object(df, index=True).values

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
            "core/schema",
            "core/data",
            "core/time",
            "core/market",
            "core/artifacts",
            "core/config",
        ]

        for root in sorted(CRITICAL_DIRS):

            if not os.path.exists(root):
                continue

            for path, dirs, files in os.walk(root):

                dirs.sort()
                files = sorted(f for f in files if f.endswith(".py"))

                for f in files:

                    full = os.path.join(path, f)

                    if os.path.islink(full):
                        raise RuntimeError(
                            f"Symlink detected in training code: {full}"
                        )

                    rel = os.path.relpath(full)
                    hasher.update(rel.encode())

                    with open(full, "rb") as fh:
                        hasher.update(fh.read())

        hasher.update(get_schema_signature().encode())
        hasher.update(platform.python_version().encode())

        return hasher.hexdigest()

    #####################################################

    @staticmethod
    def capture_environment():

        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }

    #####################################################

    @staticmethod
    def _compute_metadata_hash(metadata: dict):

        clone = dict(metadata)
        clone.pop("metadata_integrity_hash", None)

        canonical = json.dumps(
            clone,
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
        dataset_rows,
        metadata_type,
        extra_fields=None,
        feature_checksum=None,
    ):

        if dataset_rows <= 0:
            raise RuntimeError("dataset_rows must be positive.")

        MetadataManager._validate_feature_contract(
            features,
            metadata_type
        )

        MetadataManager._validate_metrics(metrics)

        MetadataManager._validate_training_window(
            training_start,
            training_end
        )

        universe_snapshot = MarketUniverse.snapshot()

        if "universe_hash" not in universe_snapshot:
            raise RuntimeError("Universe snapshot invalid.")

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
            "dataset_rows": int(dataset_rows),

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

        if feature_checksum is not None:

            if not isinstance(feature_checksum, str) or len(feature_checksum) != 64:
                raise RuntimeError("Invalid feature checksum format.")

            metadata["feature_checksum"] = feature_checksum

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

    ########################################################
    # FEATURE CHECKSUM
    ########################################################

    @staticmethod
    def fingerprint_features(features: tuple) -> str:

        canonical = json.dumps(
            list(features),
            sort_keys=False
        ).encode()

        return hashlib.sha256(canonical).hexdigest()