import json
import os
import datetime
import hashlib
import pandas as pd
import platform
import numpy as np
import tempfile

from core.schema.feature_schema import (
    get_schema_signature,
    SCHEMA_VERSION,
    MODEL_FEATURES
)

from core.market.universe import MarketUniverse


class MetadataManager:

    METADATA_VERSION = "16.0"
    MIN_TRAINING_DAYS = 120
    MIN_METADATA_BYTES = 800
    MIN_FEATURE_COUNT = 10
    MIN_DATASET_ROWS = 500
    MIN_HASH_LENGTH = 64

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
        "artifact_hash",
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
        "artifact_hash",
    }

    # =====================================================
    # CANONICAL JSON
    # =====================================================

    @staticmethod
    def _canonical_json(data: dict) -> bytes:
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":")
        ).encode()

    # =====================================================
    # FILE HASH
    # =====================================================

    @staticmethod
    def hash_file(path: str) -> str:

        if not os.path.exists(path):
            raise RuntimeError(f"Cannot hash missing file: {path}")

        if os.path.islink(path):
            raise RuntimeError(f"Symlink detected in artifact: {path}")

        hasher = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    # =====================================================
    # DATASET FINGERPRINT
    # =====================================================

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

    # =====================================================
    # TRAINING CODE FINGERPRINT
    # =====================================================

    @staticmethod
    def fingerprint_training_code():

        hasher = hashlib.sha256()

        CRITICAL_DIRS = [
            "training",
            "core/models",
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

    # =====================================================
    # FEATURE CHECKSUM
    # =====================================================

    @staticmethod
    def fingerprint_features(features: tuple) -> str:

        if tuple(features) != tuple(MODEL_FEATURES):
            raise RuntimeError("Feature list mismatch during checksum.")

        canonical = json.dumps(
            list(features),
            sort_keys=False
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    # =====================================================
    # SAVE METADATA (ATOMIC)
    # =====================================================

    @staticmethod
    def save_metadata(metadata: dict, path: str):

        if os.path.islink(path):
            raise RuntimeError("Refusing to write metadata to symlink.")

        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=directory,
            suffix=".tmp",
            encoding="utf-8"
        ) as tmp:

            json.dump(metadata, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name

        os.replace(temp_name, path)

        if os.path.getsize(path) < MetadataManager.MIN_METADATA_BYTES:
            raise RuntimeError("Metadata file below minimum size.")

    # =====================================================
    # LOAD + VERIFY
    # =====================================================

    @staticmethod
    def load_metadata(path: str) -> dict:

        if not os.path.exists(path):
            raise RuntimeError("Metadata file missing.")

        if os.path.islink(path):
            raise RuntimeError("Symlink detected in metadata file.")

        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        MetadataManager.verify_metadata_integrity(metadata)

        return metadata

    # =====================================================
    # VERIFY INTEGRITY
    # =====================================================

    @staticmethod
    def verify_metadata_integrity(metadata: dict):

        for field in MetadataManager.REQUIRED_METADATA_FIELDS:
            if field not in metadata:
                raise RuntimeError(f"Missing required metadata field: {field}")

        original_hash = metadata.get("metadata_integrity_hash")

        if not original_hash or len(original_hash) < MetadataManager.MIN_HASH_LENGTH:
            raise RuntimeError("Invalid metadata integrity hash.")

        metadata_copy = dict(metadata)
        metadata_copy.pop("metadata_integrity_hash")

        recalculated = hashlib.sha256(
            MetadataManager._canonical_json(metadata_copy)
        ).hexdigest()

        if recalculated != original_hash:
            raise RuntimeError("Metadata integrity verification failed.")

    # =====================================================
    # METRIC VALIDATION
    # =====================================================

    @staticmethod
    def _validate_metrics(metrics: dict):

        if not isinstance(metrics, dict) or not metrics:
            raise RuntimeError("Metrics must be non-empty dict.")

        for k, v in metrics.items():
            val = float(v)
            if not np.isfinite(val):
                raise RuntimeError(f"Non-finite metric detected: {k}")
            if abs(val) > 1e6:
                raise RuntimeError(f"Metric unrealistic: {k}")

    # =====================================================
    # TRAINING WINDOW VALIDATION
    # =====================================================

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

    # =====================================================
    # CREATE METADATA
    # =====================================================

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

        if dataset_rows < MetadataManager.MIN_DATASET_ROWS:
            raise RuntimeError("dataset_rows below institutional minimum.")

        if tuple(features) != tuple(MODEL_FEATURES):
            raise RuntimeError("Feature schema mismatch.")

        if len(features) < MetadataManager.MIN_FEATURE_COUNT:
            raise RuntimeError("Feature count below institutional minimum.")

        MetadataManager._validate_metrics(metrics)
        MetadataManager._validate_training_window(
            training_start,
            training_end
        )

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
            "dataset_rows": int(dataset_rows),

            "features": list(features),
            "feature_count": len(features),

            "metrics": metrics,

            "schema_signature": get_schema_signature(),
            "schema_version": SCHEMA_VERSION,

            "training_code_hash":
                MetadataManager.fingerprint_training_code(),

            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },

            "training_universe": universe_snapshot,
            "universe_hash": universe_snapshot["universe_hash"]
        }

        if feature_checksum:
            metadata["feature_checksum"] = feature_checksum

        if extra_fields:

            forbidden = MetadataManager.IMMUTABLE_KEYS & set(extra_fields)

            if forbidden:
                raise RuntimeError(
                    f"Attempt to override immutable metadata fields: {forbidden}"
                )

            metadata.update(dict(extra_fields))

        metadata["metadata_integrity_hash"] = hashlib.sha256(
            MetadataManager._canonical_json(
                {k: v for k, v in metadata.items()}
            )
        ).hexdigest()

        return metadata