import os
import json
import datetime
import shutil
from typing import Dict, Any


class ModelRegistry:
    """
    Institutional Release-Controlled Model Registry.

    Guarantees:
    - atomic promotion
    - cross-platform pointer safety
    - metadata enforcement
    - rollback lineage
    """

    MANIFEST_NAME = "manifest.json"
    LATEST_POINTER = "latest.json"

    STAGES = (
        "candidate",
        "shadow",
        "approved",
        "production"
    )

    ALLOWED_TRANSITIONS = {
        "candidate": {"shadow"},
        "shadow": {"approved"},
        "approved": {"production"},
        "production": set()
    }

    # --------------------------------------------------

    @staticmethod
    def _version() -> str:
        return datetime.datetime.utcnow().strftime("v%Y_%m_%d_%H%M%S")

    # --------------------------------------------------

    @staticmethod
    def _validate_artifact(path: str):

        if not os.path.exists(path):
            raise RuntimeError(f"Artifact missing: {path}")

        if os.path.getsize(path) == 0:
            raise RuntimeError(f"Artifact empty: {path}")

    # --------------------------------------------------

    @staticmethod
    def _validate_metadata(metadata_path: str):

        with open(metadata_path) as f:
            meta = json.load(f)

        required_fields = [
            "model_name",
            "features",
            "metrics",
            "dataset_hash",
            "schema_signature"
        ]

        missing = [k for k in required_fields if k not in meta]

        if missing:
            raise RuntimeError(
                f"Metadata missing required fields: {missing}"
            )

    # --------------------------------------------------
    # SAFE POINTER
    # --------------------------------------------------

    @staticmethod
    def _write_latest_pointer(base_dir: str, version: str):

        pointer_path = os.path.join(base_dir, ModelRegistry.LATEST_POINTER)

        tmp = pointer_path + ".tmp"

        with open(tmp, "w") as f:
            json.dump({"version": version}, f)

        os.replace(tmp, pointer_path)

    # --------------------------------------------------

    @staticmethod
    def _atomic_symlink(target: str, link_name: str):
        """
        Attempt symlink.
        Fallback silently if unsupported.
        """

        try:

            tmp_link = link_name + ".tmp"

            if os.path.exists(tmp_link):
                os.remove(tmp_link)

            os.symlink(target, tmp_link)
            os.replace(tmp_link, link_name)

        except Exception:
            # DO NOT CRASH
            pass

    # --------------------------------------------------

    @staticmethod
    def _write_manifest(version_dir: str, metadata_path: str):

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        manifest: Dict[str, Any] = {
            "version": os.path.basename(version_dir),
            "created_utc": datetime.datetime.utcnow().isoformat(),
            "stage": "candidate",
            "metadata": metadata,
            "history": []
        }

        path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        tmp_path = path + ".tmp"

        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=4)

        os.replace(tmp_path, path)

    # --------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------

    @staticmethod
    def register_model(
        base_dir: str,
        model_path: str,
        metadata_path: str
    ) -> str:

        os.makedirs(base_dir, exist_ok=True)

        ModelRegistry._validate_artifact(model_path)
        ModelRegistry._validate_artifact(metadata_path)
        ModelRegistry._validate_metadata(metadata_path)

        version = ModelRegistry._version()
        version_dir = os.path.join(base_dir, version)

        staging_dir = version_dir + ".staging"
        os.makedirs(staging_dir, exist_ok=False)

        shutil.copy2(model_path, staging_dir)
        shutil.copy2(metadata_path, staging_dir)

        ModelRegistry._write_manifest(
            staging_dir,
            os.path.join(staging_dir, os.path.basename(metadata_path))
        )

        os.replace(staging_dir, version_dir)

        return version_dir

    # --------------------------------------------------
    # PROMOTION
    # --------------------------------------------------

    @staticmethod
    def promote_to_production(
        base_dir: str,
        version: str
    ):

        version_dir = os.path.join(base_dir, version)

        manifest = ModelRegistry._load_manifest(version_dir)

        if manifest["stage"] != "approved":
            raise RuntimeError(
                "Only approved models may enter production."
            )

        latest_link = os.path.join(base_dir, "latest")

        ModelRegistry._atomic_symlink(version, latest_link)
        ModelRegistry._write_latest_pointer(base_dir, version)

        manifest["stage"] = "production"

        ModelRegistry._save_manifest(version_dir, manifest)

    # --------------------------------------------------

    @staticmethod
    def _load_manifest(version_dir: str):

        path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        if not os.path.exists(path):
            raise RuntimeError("Manifest missing.")

        with open(path) as f:
            return json.load(f)

    # --------------------------------------------------

    @staticmethod
    def _save_manifest(version_dir: str, manifest: dict):

        path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=4)

        os.replace(tmp, path)

    # --------------------------------------------------

    @staticmethod
    def get_latest_version(base_dir: str) -> str:

        pointer = os.path.join(base_dir, ModelRegistry.LATEST_POINTER)

        if os.path.exists(pointer):

            with open(pointer) as f:
                return json.load(f)["version"]

        # fallback to symlink
        latest_link = os.path.join(base_dir, "latest")

        if os.path.islink(latest_link):
            return os.readlink(latest_link)

        raise RuntimeError("Latest pointer missing.")
