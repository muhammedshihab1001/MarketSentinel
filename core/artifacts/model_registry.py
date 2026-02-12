import os
import json
import datetime
import shutil
import hashlib
import uuid
from typing import Dict, Any, List

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
    def _sha256(path: str) -> str:

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
            json.dump(payload, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

        parent = os.path.dirname(path) or "."
        ModelRegistry._fsync_dir(parent)

    ########################################################
    # LOCK (TRUE ATOMIC)
    ########################################################

    @staticmethod
    def _acquire_lock(lock_path: str):

        try:
            fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.close(fd)

        except FileExistsError:
            raise RuntimeError(
                "Registry promotion already in progress."
            )

    ########################################################

    @staticmethod
    def _release_lock(lock_path: str):

        if os.path.exists(lock_path):
            os.remove(lock_path)

    ########################################################
    # MANIFEST HASH
    ########################################################

    @staticmethod
    def _manifest_hash(manifest: dict) -> str:

        clone = dict(manifest)
        clone.pop("manifest_integrity_hash", None)

        canonical = json.dumps(
            clone,
            sort_keys=True
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    ########################################################
    # METADATA VALIDATION
    ########################################################

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

        return meta

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

        if manifest["manifest_integrity_hash"] != (
            ModelRegistry._manifest_hash(manifest)
        ):
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
                "artifacts": {
                    model_name: ModelRegistry._sha256(staged_model),
                    metadata_name: ModelRegistry._sha256(staged_meta),
                },
                "history": []
            }

            manifest["manifest_integrity_hash"] = (
                ModelRegistry._manifest_hash(manifest)
            )

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

    ########################################################
    # SELF HEAL POINTER
    ########################################################

    @staticmethod
    def _self_heal_pointer(base_dir: str) -> str:

        versions = ModelRegistry.list_versions(base_dir)

        if not versions:
            raise RuntimeError("Registry empty.")

        latest = versions[-1]

        pointer = {
            "version": latest,
            "healed": True,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

        pointer["pointer_hash"] = hashlib.sha256(
            json.dumps(pointer, sort_keys=True).encode()
        ).hexdigest()

        pointer_path = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        ModelRegistry._atomic_json_write(pointer_path, pointer)

        return latest

    ########################################################
    # PROMOTION
    ########################################################

    @staticmethod
    def promote_to_latest(base_dir: str, version: str):

        lock_path = os.path.join(base_dir, ModelRegistry.PROMOTION_LOCK)

        ModelRegistry._acquire_lock(lock_path)

        try:

            version_dir = os.path.join(base_dir, version)

            if not os.path.exists(version_dir):
                raise RuntimeError("Cannot promote missing version.")

            ModelRegistry.verify_artifacts(base_dir, version)

            pointer = {
                "version": version,
                "promoted_utc": datetime.datetime.utcnow().isoformat()
            }

            pointer["pointer_hash"] = hashlib.sha256(
                json.dumps(pointer, sort_keys=True).encode()
            ).hexdigest()

            pointer_path = os.path.join(
                base_dir,
                ModelRegistry.LATEST_POINTER
            )

            ModelRegistry._atomic_json_write(pointer_path, pointer)

        finally:
            ModelRegistry._release_lock(lock_path)

    ########################################################
    # GET LATEST (SELF HEALING)
    ########################################################

    @staticmethod
    def get_latest_version(base_dir: str) -> str:

        pointer_path = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        if not os.path.exists(pointer_path):
            return ModelRegistry._self_heal_pointer(base_dir)

        try:

            with open(pointer_path) as f:
                payload = json.load(f)

            expected = payload.get("pointer_hash")

            clone = dict(payload)
            clone.pop("pointer_hash", None)

            actual = hashlib.sha256(
                json.dumps(clone, sort_keys=True).encode()
            ).hexdigest()

            if expected != actual:
                raise RuntimeError("Pointer corrupted.")

            return payload["version"]

        except Exception:

            return ModelRegistry._self_heal_pointer(base_dir)

    ########################################################
    # LIST
    ########################################################

    @staticmethod
    def list_versions(base_dir: str) -> List[str]:

        if not os.path.exists(base_dir):
            return []

        versions = []

        for name in os.listdir(base_dir):

            path = os.path.join(base_dir, name)

            if not os.path.isdir(path):
                continue

            if name.endswith(".staging"):
                continue

            if name.startswith("."):
                continue

            manifest = os.path.join(
                path,
                ModelRegistry.MANIFEST_NAME
            )

            if os.path.exists(manifest):
                versions.append(name)

        versions.sort()

        return versions
