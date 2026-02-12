import os
import json
import datetime
import shutil
import hashlib
import uuid
import time
from typing import Dict, Any, List

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES
)


class ModelRegistry:

    MANIFEST_NAME = "manifest.json"
    LATEST_POINTER = "latest.json"
    PROMOTION_LOCK = ".promotion.lock"

    ########################################################
    # MULTI-MANIFEST GOVERNANCE (FAIL CLOSED)
    ########################################################

    ALLOWED_METADATA_TYPES = {
        "training_manifest_v1",
        "timeseries_manifest_v1",
        "sequence_manifest_v1"
    }

    REQUIRED_METADATA_FIELDS = (
        "model_name",
        "features",
        "metrics",
        "dataset_hash",
        "schema_signature",
        "metadata_type",
        "training_code_hash",
        "metadata_integrity_hash"
    )

    LOCK_TIMEOUT_SECONDS = 600

    ########################################################
    # VERSION
    ########################################################

    @staticmethod
    def _version() -> str:
        ts = datetime.datetime.utcnow().strftime("v%Y_%m_%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{ts}_{suffix}"

    ########################################################
    # HASHING
    ########################################################

    @staticmethod
    def _sha256(path: str):
        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    ########################################################
    # FSYNC
    ########################################################

    @staticmethod
    def _fsync_dir(path: str):
        if os.name == "nt":
            return

        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    ########################################################
    # ATOMIC JSON
    ########################################################

    @staticmethod
    def _atomic_json_write(path: str, payload: dict):

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

        parent = os.path.dirname(path) or "."
        ModelRegistry._fsync_dir(parent)

    ########################################################
    # FEATURE CONTRACT
    ########################################################

    @staticmethod
    def _validate_feature_contract(meta):

        features = meta["features"]
        mtype = meta["metadata_type"]

        if mtype == "training_manifest_v1":

            if list(features) != list(MODEL_FEATURES):
                raise RuntimeError("Feature ordering mismatch.")

        elif mtype == "timeseries_manifest_v1":

            if features != ["close"]:
                raise RuntimeError(
                    "Timeseries models must declare ['close']."
                )

        elif mtype == "sequence_manifest_v1":

            if features != ["close_sequence"]:
                raise RuntimeError(
                    "Sequence models must declare ['close_sequence']."
                )

        else:
            raise RuntimeError(
                "Unknown metadata_type — refusing registry write."
            )

    ########################################################
    # METADATA VALIDATION
    ########################################################

    @staticmethod
    def _metadata_hash(meta: dict):

        clone = dict(meta)
        clone.pop("metadata_integrity_hash", None)

        canonical = json.dumps(
            clone,
            sort_keys=True
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _validate_metadata(metadata_path: str):

        with open(metadata_path) as f:
            meta = json.load(f)

        missing = [
            k for k in ModelRegistry.REQUIRED_METADATA_FIELDS
            if k not in meta
        ]

        if missing:
            raise RuntimeError(
                f"Metadata missing required fields: {missing}"
            )

        if meta["metadata_type"] not in ModelRegistry.ALLOWED_METADATA_TYPES:
            raise RuntimeError(
                "Invalid metadata_type — refusing registry write."
            )

        if meta["metadata_integrity_hash"] != ModelRegistry._metadata_hash(meta):
            raise RuntimeError("Metadata integrity failure.")

        if meta["schema_signature"] != get_schema_signature():
            raise RuntimeError("Schema mismatch detected.")

        ModelRegistry._validate_feature_contract(meta)

        return meta

    ########################################################
    # LOCK WITH DURABILITY
    ########################################################

    @staticmethod
    def _acquire_lock(lock_path: str):

        directory = os.path.dirname(lock_path) or "."

        if os.path.exists(lock_path):

            age = time.time() - os.path.getmtime(lock_path)

            if age > ModelRegistry.LOCK_TIMEOUT_SECONDS:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass

                if os.path.exists(lock_path):
                    raise RuntimeError(
                        "Failed to clear stale promotion lock."
                    )
            else:
                raise RuntimeError(
                    "Registry promotion already in progress."
                )

        fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
        os.close(fd)

        ModelRegistry._fsync_dir(directory)

    ########################################################

    @staticmethod
    def _release_lock(lock_path: str):

        directory = os.path.dirname(lock_path) or "."

        if os.path.exists(lock_path):
            os.remove(lock_path)
            ModelRegistry._fsync_dir(directory)

    ########################################################
    # VERIFY ARTIFACTS
    ########################################################

    @staticmethod
    def verify_artifacts(base_dir: str, version: str):

        version_dir = os.path.join(base_dir, version)

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        if not os.path.exists(manifest_path):
            raise RuntimeError("Manifest missing.")

        with open(manifest_path) as f:
            manifest = json.load(f)

        clone = dict(manifest)
        expected_hash = clone.pop("manifest_integrity_hash")

        actual_hash = hashlib.sha256(
            json.dumps(clone, sort_keys=True).encode()
        ).hexdigest()

        if expected_hash != actual_hash:
            raise RuntimeError("Manifest integrity failure.")

        for artifact, expected_hash in manifest["artifacts"].items():

            artifact_path = os.path.join(version_dir, artifact)

            if not os.path.exists(artifact_path):
                raise RuntimeError(f"Artifact missing: {artifact}")

            actual = ModelRegistry._sha256(artifact_path)

            if actual != expected_hash:
                raise RuntimeError(
                    f"Artifact integrity failure: {artifact}"
                )

    ########################################################
    # REGISTER MODEL
    ########################################################

    @staticmethod
    def register_model(
        base_dir: str,
        model_path: str,
        metadata_path: str,
        parent_version: str | None = None
    ) -> str:

        os.makedirs(base_dir, exist_ok=True)

        meta = ModelRegistry._validate_metadata(metadata_path)

        version = ModelRegistry._version()

        version_dir = os.path.join(base_dir, version)
        staging_dir = version_dir + ".staging"

        os.makedirs(staging_dir, exist_ok=False)

        try:

            model_name = os.path.basename(model_path)
            metadata_name = os.path.basename(metadata_path)

            staged_model = os.path.join(staging_dir, model_name)
            staged_meta = os.path.join(staging_dir, metadata_name)

            shutil.copy2(model_path, staged_model)
            shutil.copy2(metadata_path, staged_meta)

            manifest: Dict[str, Any] = {
                "version": version,
                "created_utc": datetime.datetime.utcnow().isoformat(),
                "stage": "candidate",
                "parent": parent_version,
                "dataset_hash": meta["dataset_hash"],
                "schema_signature": meta["schema_signature"],
                "training_code_hash": meta["training_code_hash"],
                "metadata_type": meta["metadata_type"],
                "artifacts": {
                    model_name: ModelRegistry._sha256(staged_model),
                    metadata_name: ModelRegistry._sha256(staged_meta),
                },
                "history": []
            }

            clone = dict(manifest)

            manifest["manifest_integrity_hash"] = hashlib.sha256(
                json.dumps(clone, sort_keys=True).encode()
            ).hexdigest()

            manifest_path = os.path.join(
                staging_dir,
                ModelRegistry.MANIFEST_NAME
            )

            ModelRegistry._atomic_json_write(
                manifest_path,
                manifest
            )

            os.replace(staging_dir, version_dir)

            ModelRegistry._fsync_dir(base_dir)
            ModelRegistry._fsync_dir(version_dir)

            ModelRegistry.verify_artifacts(base_dir, version)

            return version

        except Exception:

            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)

            raise
