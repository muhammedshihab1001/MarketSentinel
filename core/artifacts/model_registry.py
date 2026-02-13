import os
import json
import datetime
import shutil
import hashlib
import uuid
from typing import Dict, Any

from core.schema.feature_schema import get_schema_signature
from core.artifacts.metadata_manager import MetadataManager


class ModelRegistry:

    MANIFEST_NAME = "manifest.json"
    LATEST_POINTER = "latest.json"
    PROMOTION_LOCK = ".promotion.lock"

    LOCK_TIMEOUT_SECONDS = 600

    ########################################################

    @staticmethod
    def _version() -> str:
        ts = datetime.datetime.utcnow().strftime("v%Y_%m_%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{ts}_{suffix}"

    ########################################################

    @staticmethod
    def _sha256(path: str):

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

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

    @staticmethod
    def _normalize_for_hash(obj):

        if isinstance(obj, dict):
            return {k: ModelRegistry._normalize_for_hash(v)
                    for k, v in sorted(obj.items())}

        if isinstance(obj, list):
            return [ModelRegistry._normalize_for_hash(v) for v in obj]

        return obj

    ########################################################

    @staticmethod
    def _atomic_json_write(path: str, payload: dict):

        tmp = path + ".tmp"

        try:

            with open(tmp, "w") as f:
                json.dump(payload, f, indent=4, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, path)

            parent = os.path.dirname(path) or "."
            ModelRegistry._fsync_dir(parent)

        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    ########################################################

    @staticmethod
    def _manifest_hash(manifest: dict):

        clone = dict(manifest)
        clone.pop("manifest_integrity_hash", None)

        normalized = ModelRegistry._normalize_for_hash(clone)

        canonical = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":")
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    ########################################################

    @staticmethod
    def _validate_parent(base_dir: str, parent_version: str, meta: dict):

        parent_manifest_path = os.path.join(
            base_dir,
            parent_version,
            ModelRegistry.MANIFEST_NAME
        )

        if not os.path.exists(parent_manifest_path):
            raise RuntimeError("Parent manifest not found.")

        with open(parent_manifest_path) as f:
            parent = json.load(f)

        if parent["schema_signature"] != meta["schema_signature"]:
            raise RuntimeError(
                "Parent schema mismatch detected."
            )

        if parent.get("universe_hash") != \
           meta["training_universe"]["universe_hash"]:
            raise RuntimeError(
                "Parent universe mismatch detected."
            )

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

    @staticmethod
    def register_model(
        base_dir: str,
        model_path: str,
        metadata_path: str,
        parent_version: str | None = None
    ) -> str:

        os.makedirs(base_dir, exist_ok=True)

        meta = MetadataManager.load_metadata(metadata_path)

        if parent_version:
            ModelRegistry._validate_parent(
                base_dir,
                parent_version,
                meta
            )

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
                "training_window": meta["training_window"],

                "universe_hash":
                    meta["training_universe"]["universe_hash"],

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
