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

    ########################################################
    # VERSION
    ########################################################

    @staticmethod
    def _version() -> str:
        ts = datetime.datetime.utcnow().strftime("v%Y_%m_%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{ts}_{suffix}"

    ########################################################
    # DIRECTORY HASH (DETERMINISTIC)
    ########################################################

    @staticmethod
    def _hash_directory(path: str) -> str:

        hasher = hashlib.sha256()

        for root, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()

            for f in files:
                file_path = os.path.join(root, f)

                rel = os.path.relpath(file_path, path)
                hasher.update(rel.encode())

                with open(file_path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        hasher.update(chunk)

        return hasher.hexdigest()

    ########################################################
    # FILE OR DIRECTORY HASH
    ########################################################

    @staticmethod
    def _sha256(path: str) -> str:

        if os.path.isdir(path):
            return ModelRegistry._hash_directory(path)

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    ########################################################
    # FSYNC (SAFE ON LINUX, NO-OP ON WINDOWS)
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
    # ATOMIC JSON WRITE
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
    # MANIFEST VALIDATION
    ########################################################

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

        metadata_type = meta.get("metadata_type")

        # Strict ordering ONLY for tabular models
        if metadata_type in ("tabular", "tabular_model"):

            if meta["features"] != list(MODEL_FEATURES):
                raise RuntimeError("Feature ordering mismatch.")

    ########################################################
    # SAFE COPY
    ########################################################

    @staticmethod
    def _copy_artifact(src: str, dst: str):

        if not os.path.exists(src):
            raise RuntimeError(f"Artifact missing before copy: {src}")

        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

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

        ModelRegistry._validate_metadata(metadata_path)

        version = ModelRegistry._version()

        version_dir = os.path.join(base_dir, version)
        staging_dir = version_dir + ".staging"

        os.makedirs(staging_dir, exist_ok=False)

        try:

            model_name = os.path.basename(model_path)
            metadata_name = os.path.basename(metadata_path)

            staged_model = os.path.join(staging_dir, model_name)
            staged_meta = os.path.join(staging_dir, metadata_name)

            ModelRegistry._copy_artifact(model_path, staged_model)
            ModelRegistry._copy_artifact(metadata_path, staged_meta)

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

        except Exception:
            # Clean failed staging
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise
