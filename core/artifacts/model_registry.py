import os
import json
import datetime
import shutil
import hashlib
import uuid
from typing import Dict, Any

from core.schema.feature_schema import get_schema_signature


class ModelRegistry:
    """
    Institutional Transactional Model Registry.

    Guarantees:
    - atomic registration
    - artifact immutability
    - pointer crash safety
    - promotion governance
    - rollback protection
    - lineage tracking
    - schema enforcement
    - artifact integrity verification
    """

    MANIFEST_NAME = "manifest.json"
    LATEST_POINTER = "latest.json"

    STAGES = ("candidate", "shadow", "approved", "production")

    ALLOWED_TRANSITIONS = {
        "candidate": {"shadow"},
        "shadow": {"approved"},
        "approved": {"production"},
        "production": set()
    }

    REQUIRED_METADATA_FIELDS = (
        "model_name",
        "features",
        "metrics",
        "dataset_hash",
        "schema_signature",
        "metadata_type",
        "training_code_hash"
    )

    # --------------------------------------------------

    @staticmethod
    def _version() -> str:
        ts = datetime.datetime.utcnow().strftime("v%Y_%m_%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{ts}_{suffix}"

    # --------------------------------------------------

    @staticmethod
    def _sha256(path: str) -> str:

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    # --------------------------------------------------

    @staticmethod
    def _fsync_dir(path: str):
        fd = os.open(path, os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)

    # --------------------------------------------------

    @staticmethod
    def _atomic_json_write(path: str, payload: dict):

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    # --------------------------------------------------

    @staticmethod
    def _validate_metadata(metadata_path: str):

        with open(metadata_path) as f:
            meta = json.load(f)

        if meta.get("metadata_type") != "model":
            raise RuntimeError("Attempted to register non-model metadata.")

        missing = [
            k for k in ModelRegistry.REQUIRED_METADATA_FIELDS
            if k not in meta
        ]

        if missing:
            raise RuntimeError(
                f"Metadata missing required fields: {missing}"
            )

        # HARD SCHEMA LOCK
        if meta["schema_signature"] != get_schema_signature():
            raise RuntimeError(
                "Schema mismatch detected. Refusing registry."
            )

    # --------------------------------------------------
    # ARTIFACT VERIFICATION
    # --------------------------------------------------

    @staticmethod
    def verify_artifacts(base_dir: str, version: str):

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError(
                "Artifact verification failed. Version directory missing."
            )

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        if not os.path.exists(manifest_path):
            raise RuntimeError("Manifest missing during verification.")

        with open(manifest_path) as f:
            manifest = json.load(f)

        for artifact, expected_hash in manifest["artifacts"].items():

            artifact_path = os.path.join(version_dir, artifact)

            if not os.path.exists(artifact_path):
                raise RuntimeError(
                    f"Artifact missing: {artifact}"
                )

            actual = ModelRegistry._sha256(artifact_path)

            if actual != expected_hash:
                raise RuntimeError(
                    f"Artifact integrity failure: {artifact}"
                )

    # --------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------

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

    # --------------------------------------------------
    # PROMOTION
    # --------------------------------------------------

    @staticmethod
    def promote_to_production(base_dir: str, version: str):

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError("Version directory missing.")

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        if not os.path.exists(manifest_path):
            raise RuntimeError("Manifest missing.")

        with open(manifest_path) as f:
            manifest = json.load(f)

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

        latest_link = os.path.join(base_dir, "latest")

        try:
            tmp = latest_link + ".tmp"
            os.symlink(version, tmp)
            os.replace(tmp, latest_link)
        except Exception:
            pass

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

    # --------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------

    @staticmethod
    def rollback(base_dir: str, version: str):

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError(
                "Rollback failed. Version directory missing."
            )

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        if not os.path.exists(manifest_path):
            raise RuntimeError(
                "Rollback failed. Manifest missing."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        if manifest["stage"] != "production":
            raise RuntimeError(
                "Rollback allowed only to production models."
            )

        pointer_path = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        ModelRegistry._atomic_json_write(
            pointer_path,
            {"version": version}
        )

    # --------------------------------------------------

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
                "Registry pointer corrupted. Version directory missing."
            )

        return version
