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
    - metadata contract enforcement
    - schema protection
    - pointer safety
    - rollback lineage
    """

    MANIFEST_NAME = "manifest.json"

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
            "schema_version",
            "schema_signature"
        ]

        missing = [k for k in required_fields if k not in meta]

        if missing:
            raise RuntimeError(
                f"Metadata missing required fields: {missing}"
            )

        if not isinstance(meta["schema_signature"], str):
            raise RuntimeError("schema_signature must be a string")

        if not isinstance(meta["metrics"], dict):
            raise RuntimeError("metrics must be a dictionary")

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

        manifest = ModelRegistry._load_manifest(version_dir)

        current_stage = manifest["stage"]

        if new_stage not in ModelRegistry.ALLOWED_TRANSITIONS[current_stage]:
            raise RuntimeError(
                f"Illegal transition {current_stage} -> {new_stage}"
            )

        manifest["history"].append({
            "from": current_stage,
            "to": new_stage,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

        manifest["stage"] = new_stage

        ModelRegistry._save_manifest(version_dir, manifest)

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

        latest_path = os.path.join(base_dir, "latest")

        previous = None
        if os.path.islink(latest_path):
            previous = os.readlink(latest_path)

        ModelRegistry._atomic_symlink(version, latest_path)

        manifest["history"].append({
            "from": "approved",
            "to": "production",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "previous_production": previous
        })

        manifest["stage"] = "production"

        ModelRegistry._save_manifest(version_dir, manifest)

    # --------------------------------------------------

    @staticmethod
    def rollback(base_dir: str, version: str):

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError("Rollback failed — version not found.")

        latest_path = os.path.join(base_dir, "latest")

        previous = None
        if os.path.islink(latest_path):
            previous = os.readlink(latest_path)

        ModelRegistry._atomic_symlink(version, latest_path)

        manifest = ModelRegistry._load_manifest(version_dir)

        manifest["history"].append({
            "event": "rollback",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "replaced_version": previous
        })

        manifest["stage"] = "production"

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

        return version
