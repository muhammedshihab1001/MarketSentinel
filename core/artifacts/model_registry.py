import os
import json
import datetime
import shutil
from typing import Dict, Any


class ModelRegistry:
    """
    Institutional Release-Controlled Model Registry.

    Guarantees:
    - zero artifact mutation
    - atomic registration
    - pointer safety
    - promotion lineage
    - rollback capability
    - cross-platform behavior
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

    REQUIRED_METADATA_FIELDS = (
        "model_name",
        "features",
        "metrics",
        "dataset_hash",
        "schema_signature"
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
    # STRICT METADATA VALIDATION (NO MUTATION)
    # --------------------------------------------------

    @staticmethod
    def _validate_metadata(metadata_path: str):

        with open(metadata_path) as f:
            meta = json.load(f)

        if meta.get("metadata_type", "model") != "model":
            raise RuntimeError(
                "Attempted to register non-model metadata."
            )

        missing = [
            k for k in ModelRegistry.REQUIRED_METADATA_FIELDS
            if k not in meta
        ]

        if missing:
            raise RuntimeError(
                f"Metadata missing required fields: {missing}"
            )

        if not isinstance(meta["features"], list):
            raise RuntimeError("features must be a list")

        if not isinstance(meta["metrics"], dict):
            raise RuntimeError("metrics must be a dictionary")

        if not isinstance(meta["dataset_hash"], str):
            raise RuntimeError("dataset_hash must be a string")

    # --------------------------------------------------
    # POINTER
    # --------------------------------------------------

    @staticmethod
    def _write_latest_pointer(base_dir: str, version: str):

        pointer_path = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        tmp = pointer_path + ".tmp"

        with open(tmp, "w") as f:
            json.dump({"version": version}, f)

        os.replace(tmp, pointer_path)

    # --------------------------------------------------

    @staticmethod
    def _atomic_symlink(target: str, link_name: str):

        try:

            tmp_link = link_name + ".tmp"

            if os.path.exists(tmp_link):
                os.remove(tmp_link)

            os.symlink(target, tmp_link)
            os.replace(tmp_link, link_name)

        except Exception:
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

        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=4)

        os.replace(tmp, path)

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
            os.path.join(
                staging_dir,
                os.path.basename(metadata_path)
            )
        )

        os.replace(staging_dir, version_dir)

        return version

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

        latest_link = os.path.join(base_dir, "latest")

        previous = None

        if os.path.exists(ModelRegistry.LATEST_POINTER):
            try:
                previous = ModelRegistry.get_latest_version(base_dir)
            except Exception:
                pass

        ModelRegistry._atomic_symlink(version, latest_link)
        ModelRegistry._write_latest_pointer(base_dir, version)

        manifest["history"].append({
            "event": "promotion",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "previous_production": previous
        })

        manifest["stage"] = "production"

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

        ModelRegistry._write_latest_pointer(
            base_dir,
            version
        )

    # --------------------------------------------------

    @staticmethod
    def get_latest_version(base_dir: str) -> str:

        pointer = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        if os.path.exists(pointer):

            with open(pointer) as f:
                return json.load(f)["version"]

        latest_link = os.path.join(base_dir, "latest")

        if os.path.islink(latest_link):
            return os.readlink(latest_link)

        raise RuntimeError("Latest pointer missing.")
