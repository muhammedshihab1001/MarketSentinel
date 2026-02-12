import os
import json
import datetime
import shutil
import hashlib
import uuid
import time
from typing import Dict, Any

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES
)


class ModelRegistry:

    MANIFEST_NAME = "manifest.json"
    LATEST_POINTER = "latest.json"
    PROMOTION_LOCK = ".promotion.lock"

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
    # HASH HELPER (⭐ NEW)
    ########################################################

    @staticmethod
    def _hash_list(items):
        return hashlib.sha256(
            json.dumps(sorted(items)).encode()
        ).hexdigest()

    ########################################################
    # METADATA VALIDATION (UPGRADED)
    ########################################################

    @staticmethod
    def _metadata_hash(meta: dict):

        clone = dict(meta)
        clone.pop("metadata_integrity_hash", None)

        return hashlib.sha256(
            json.dumps(clone, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _validate_metadata(metadata_path: str):

        with open(metadata_path) as f:
            meta = json.load(f)

        # integrity
        if meta["metadata_integrity_hash"] != ModelRegistry._metadata_hash(meta):
            raise RuntimeError("Metadata integrity failure.")

        # schema
        if meta["schema_signature"] != get_schema_signature():
            raise RuntimeError("Schema mismatch detected.")

        # feature contract
        if meta["metadata_type"] == "training_manifest_v1":

            if list(meta["features"]) != list(MODEL_FEATURES):
                raise RuntimeError("Feature ordering mismatch.")

        # ⭐ CRITICAL — training window required
        if "training_window" not in meta:
            raise RuntimeError(
                "Metadata missing training_window — refusing registry write."
            )

        return meta

    ########################################################
    # MANIFEST HASH
    ########################################################

    @staticmethod
    def _manifest_hash(manifest: dict):

        clone = dict(manifest)
        clone.pop("manifest_integrity_hash", None)

        return hashlib.sha256(
            json.dumps(clone, sort_keys=True).encode()
        ).hexdigest()

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
    # REGISTER MODEL (INSTITUTIONAL)
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

            ##################################################
            # ⭐ UNIVERSE HASH (VERY IMPORTANT)
            ##################################################

            universe_hash = None

            if "training_universe" in meta:
                universe_hash = ModelRegistry._hash_list(
                    meta["training_universe"]
                )

            ##################################################

            manifest: Dict[str, Any] = {

                "version": version,
                "created_utc": datetime.datetime.utcnow().isoformat(),
                "stage": "candidate",
                "parent": parent_version,

                "dataset_hash": meta["dataset_hash"],
                "schema_signature": meta["schema_signature"],
                "training_code_hash": meta["training_code_hash"],
                "training_window": meta["training_window"],

                "universe_hash": universe_hash,
                "metadata_type": meta["metadata_type"],

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
