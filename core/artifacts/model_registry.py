import os
import json
import datetime
import shutil
from typing import Dict, Any


class ModelRegistry:
    """
    Transactional filesystem model registry.

    Guarantees:
    - atomic promotion
    - pointer safety
    - rollback integrity
    - artifact validation
    - version manifest
    """

    MANIFEST_NAME = "manifest.json"

    # --------------------------------------------------

    @staticmethod
    def _version() -> str:
        return datetime.datetime.utcnow().strftime("v%Y_%m_%d_%H%M%S")

    # --------------------------------------------------
    # SAFETY VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _validate_artifact(path: str):

        if not os.path.exists(path):
            raise RuntimeError(f"Artifact missing: {path}")

        if os.path.getsize(path) == 0:
            raise RuntimeError(f"Artifact empty: {path}")

    # --------------------------------------------------
    # ATOMIC POINTER SWAP
    # --------------------------------------------------

    @staticmethod
    def _atomic_symlink(target: str, link_name: str):

        tmp_link = link_name + ".tmp"

        if os.path.exists(tmp_link):
            os.remove(tmp_link)

        os.symlink(target, tmp_link)

        # atomic replace
        os.replace(tmp_link, link_name)

    # --------------------------------------------------
    # MANIFEST CREATION
    # --------------------------------------------------

    @staticmethod
    def _write_manifest(version_dir: str, metadata_path: str):

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        manifest: Dict[str, Any] = {
            "version": os.path.basename(version_dir),
            "created_utc": datetime.datetime.utcnow().isoformat(),
            "metadata": metadata
        }

        manifest_path = os.path.join(version_dir, ModelRegistry.MANIFEST_NAME)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

    # --------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------

    @staticmethod
    def register_model(
        base_dir: str,
        model_path: str,
        metadata_path: str
    ) -> str:
        """
        Transactionally registers a model version.

        Flow:

        validate → stage → manifest → promote pointer
        """

        os.makedirs(base_dir, exist_ok=True)

        ModelRegistry._validate_artifact(model_path)
        ModelRegistry._validate_artifact(metadata_path)

        version = ModelRegistry._version()
        version_dir = os.path.join(base_dir, version)

        if os.path.exists(version_dir):
            raise RuntimeError("Version collision detected.")

        # -----------------------------
        # STAGING
        # -----------------------------

        staging_dir = version_dir + ".staging"
        os.makedirs(staging_dir, exist_ok=False)

        staged_model = os.path.join(
            staging_dir,
            os.path.basename(model_path)
        )

        staged_meta = os.path.join(
            staging_dir,
            os.path.basename(metadata_path)
        )

        shutil.copy2(model_path, staged_model)
        shutil.copy2(metadata_path, staged_meta)

        # verify staged artifacts
        ModelRegistry._validate_artifact(staged_model)
        ModelRegistry._validate_artifact(staged_meta)

        # write manifest BEFORE promotion
        ModelRegistry._write_manifest(staging_dir, staged_meta)

        # atomic directory promote
        os.replace(staging_dir, version_dir)

        # -----------------------------
        # POINTER UPDATE
        # -----------------------------

        latest_path = os.path.join(base_dir, "latest")

        ModelRegistry._atomic_symlink(version, latest_path)

        return version_dir

    # --------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------

    @staticmethod
    def rollback(base_dir: str, version: str):
        """
        Instantly repoints 'latest' to a previous version.
        """

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError("Rollback failed — version not found.")

        latest_path = os.path.join(base_dir, "latest")

        ModelRegistry._atomic_symlink(version, latest_path)

    # --------------------------------------------------
    # LOAD LATEST
    # --------------------------------------------------

    @staticmethod
    def get_latest_version(base_dir: str) -> str:

        latest_path = os.path.join(base_dir, "latest")

        if not os.path.islink(latest_path):
            raise RuntimeError("Latest pointer missing.")

        version = os.readlink(latest_path)

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError("Latest pointer corrupted.")

        return version_dir
