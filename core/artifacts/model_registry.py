import os
import json
import datetime
import shutil
import hashlib
import uuid
from typing import Dict, Any

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES
)


class ModelRegistry:

    MANIFEST_NAME = "manifest.json"
    LATEST_POINTER = "latest.json"
    PROMOTION_LOCK = ".promotion.lock"

    REQUIRED_METADATA_FIELDS = (
        "model_name",
        "features",
        "metrics",
        "dataset_hash",
        "schema_signature",
        "metadata_type",
        "training_code_hash"
    )

    @staticmethod
    def _version() -> str:
        ts = datetime.datetime.utcnow().strftime("v%Y_%m_%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{ts}_{suffix}"

    @staticmethod
    def _sha256(path: str) -> str:

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    @staticmethod
    def _fsync_dir(path: str):
        fd = os.open(path, os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)

    @staticmethod
    def _atomic_json_write(path: str, payload: dict):

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

        parent = os.path.dirname(path) or "."
        ModelRegistry._fsync_dir(parent)

    @staticmethod
    def _validate_manifest(manifest: dict):

        required = ["version", "stage", "artifacts", "history"]

        for r in required:
            if r not in manifest:
                raise RuntimeError("Manifest corrupted.")

        if not isinstance(manifest["artifacts"], dict):
            raise RuntimeError("Manifest artifacts invalid.")

        if not manifest["artifacts"]:
            raise RuntimeError("Manifest contains no artifacts.")

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

        if meta["schema_signature"] != get_schema_signature():
            raise RuntimeError("Schema mismatch detected.")

        if meta["features"] != list(MODEL_FEATURES):
            raise RuntimeError("Feature ordering mismatch.")

    @staticmethod
    def verify_artifacts(base_dir: str, version: str):

        version_dir = os.path.join(base_dir, version)

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        if not os.path.exists(manifest_path):
            raise RuntimeError("Manifest missing during verification.")

        with open(manifest_path) as f:
            manifest = json.load(f)

        ModelRegistry._validate_manifest(manifest)

        for artifact, expected_hash in manifest["artifacts"].items():

            artifact_path = os.path.join(version_dir, artifact)

            if not os.path.exists(artifact_path):
                raise RuntimeError(f"Artifact missing: {artifact}")

            actual = ModelRegistry._sha256(artifact_path)

            if actual != expected_hash:
                raise RuntimeError(
                    f"Artifact integrity failure: {artifact}"
                )

    @staticmethod
    def _acquire_lock(lock_path: str):

        try:
            fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.close(fd)

        except FileExistsError:
            raise RuntimeError("Another promotion is in progress.")

    @staticmethod
    def _release_lock(lock_path: str):
        if os.path.exists(lock_path):
            os.remove(lock_path)

    @staticmethod
    def register_model(
        base_dir: str,
        model_path: str,
        metadata_path: str,
        parent_version: str | None = None
    ) -> str:

        os.makedirs(base_dir, exist_ok=True)

        ModelRegistry._validate_metadata(metadata_path)

        version = ModelRegistry._version()

        version_dir = os.path.join(base_dir, version)
        staging_dir = version_dir + ".staging"

        os.makedirs(staging_dir, exist_ok=False)

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
            "artifacts": {
                model_name: ModelRegistry._sha256(staged_model),
                metadata_name: ModelRegistry._sha256(staged_meta),
            },
            "history": []
        }

        manifest_path = os.path.join(
            staging_dir,
            ModelRegistry.MANIFEST_NAME
        )

        ModelRegistry._atomic_json_write(manifest_path, manifest)

        os.replace(staging_dir, version_dir)
        ModelRegistry._fsync_dir(base_dir)

        return version

    @staticmethod
    def promote_to_production(base_dir: str, version: str):

        lock_path = os.path.join(base_dir, ModelRegistry.PROMOTION_LOCK)

        ModelRegistry._acquire_lock(lock_path)

        try:

            ModelRegistry.verify_artifacts(base_dir, version)

            version_dir = os.path.join(base_dir, version)

            manifest_path = os.path.join(
                version_dir,
                ModelRegistry.MANIFEST_NAME
            )

            with open(manifest_path) as f:
                manifest = json.load(f)

            ModelRegistry._validate_manifest(manifest)

            if manifest["stage"] != "approved":
                raise RuntimeError(
                    "Only approved models may enter production."
                )

            pointer_path = os.path.join(
                base_dir,
                ModelRegistry.LATEST_POINTER
            )

            previous = None

            if os.path.exists(pointer_path):
                with open(pointer_path) as f:
                    previous = json.load(f).get("version")

            ModelRegistry._atomic_json_write(
                pointer_path,
                {"version": version}
            )

            ModelRegistry._fsync_dir(base_dir)

            manifest["stage"] = "production"
            manifest["history"].append({
                "event": "promotion",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "previous": previous
            })

            ModelRegistry._atomic_json_write(
                manifest_path,
                manifest
            )

        finally:
            ModelRegistry._release_lock(lock_path)

    @staticmethod
    def rollback(base_dir: str, version: str):

        ModelRegistry.verify_artifacts(base_dir, version)

        pointer_path = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        ModelRegistry._atomic_json_write(
            pointer_path,
            {"version": version}
        )

        ModelRegistry._fsync_dir(base_dir)

    @staticmethod
    def get_latest_version(base_dir: str) -> str:

        pointer = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        if not os.path.exists(pointer):
            raise RuntimeError("Latest pointer missing.")

        with open(pointer) as f:
            version = json.load(f)["version"]

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError(
                "Registry pointer corrupted."
            )

        return version
