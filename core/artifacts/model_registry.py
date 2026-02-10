import os
import json
import datetime
import shutil
from typing import Dict, Any


class ModelRegistry:
    """
    Transactional Release-Controlled Model Registry.

    Guarantees:
    - atomic promotion
    - release stages
    - pointer safety
    - rollback integrity
    - artifact validation
    - manifest lineage
    """

    MANIFEST_NAME = "manifest.json"

    STAGES = (
        "candidate",
        "shadow",
        "approved",
        "production"
    )

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
    def _atomic_symlink(target: str, link_name: str):

        tmp_link = link_name + ".tmp"

        if os.path.exists(tmp_link):
            os.remove(tmp_link)

        os.symlink(target, tmp_link)
        os.replace(tmp_link, link_name)

    # --------------------------------------------------

    @staticmethod
    def _write_manifest(version_dir: str, metadata_path: str):

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        manifest: Dict[str, Any] = {
            "version": os.path.basename(version_dir),
            "created_utc": datetime.datetime.utcnow().isoformat(),
            "stage": "candidate",
            "metadata": metadata
        }

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

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

        with open(path, "w") as f:
            json.dump(manifest, f, indent=4)

    # --------------------------------------------------
    # REGISTRATION (CANDIDATE ONLY)
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

        version = ModelRegistry._version()
        version_dir = os.path.join(base_dir, version)

        if os.path.exists(version_dir):
            raise RuntimeError("Version collision detected.")

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

        ModelRegistry._validate_artifact(staged_model)
        ModelRegistry._validate_artifact(staged_meta)

        ModelRegistry._write_manifest(
            staging_dir,
            staged_meta
        )

        os.replace(staging_dir, version_dir)

        # DO NOT update production pointer here
        return version_dir

    # --------------------------------------------------
    # STAGE TRANSITION
    # --------------------------------------------------

    @staticmethod
    def transition_stage(
        base_dir: str,
        version: str,
        new_stage: str
    ):

        if new_stage not in ModelRegistry.STAGES:
            raise RuntimeError("Invalid stage.")

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError("Version not found.")

        manifest = ModelRegistry._load_manifest(version_dir)

        manifest["stage"] = new_stage
        manifest["stage_updated_utc"] = (
            datetime.datetime.utcnow().isoformat()
        )

        ModelRegistry._save_manifest(version_dir, manifest)

    # --------------------------------------------------
    # PROMOTE TO PRODUCTION
    # --------------------------------------------------

    @staticmethod
    def promote_to_production(
        base_dir: str,
        version: str
    ):
        """
        Explicit promotion gate.
        """

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError("Version not found.")

        manifest = ModelRegistry._load_manifest(version_dir)

        if manifest["stage"] not in ("approved", "shadow"):
            raise RuntimeError(
                "Model must be approved before production."
            )

        latest_path = os.path.join(base_dir, "latest")

        ModelRegistry._atomic_symlink(
            version,
            latest_path
        )

        manifest["stage"] = "production"
        manifest["promoted_utc"] = (
            datetime.datetime.utcnow().isoformat()
        )

        ModelRegistry._save_manifest(version_dir, manifest)

    # --------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------

    @staticmethod
    def rollback(base_dir: str, version: str):

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError("Rollback failed — version not found.")

        ModelRegistry._atomic_symlink(
            version,
            os.path.join(base_dir, "latest")
        )

        manifest = ModelRegistry._load_manifest(version_dir)
        manifest["stage"] = "production"
        manifest["rollback_utc"] = (
            datetime.datetime.utcnow().isoformat()
        )

        ModelRegistry._save_manifest(version_dir, manifest)

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
