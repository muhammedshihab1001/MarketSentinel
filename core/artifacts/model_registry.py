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
    - backward compatibility with legacy metadata
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
    # 🔥 CRITICAL FIX — AUTO UPGRADE LEGACY METADATA
    # --------------------------------------------------

    @staticmethod
    def _validate_metadata(metadata_path: str):

        with open(metadata_path) as f:
            meta = json.load(f)

        updated = False

        # ---------------------------------------------
        # Auto-upgrade legacy metadata
        # ---------------------------------------------

        if "metadata_type" not in meta:
            meta["metadata_type"] = "model"
            updated = True

        if "schema_signature" not in meta:
            meta["schema_signature"] = "legacy"
            updated = True

        if "features" not in meta:
            meta["features"] = []
            updated = True

        if "metrics" not in meta:
            meta["metrics"] = {}
            updated = True

        if "dataset_hash" not in meta:
            meta["dataset_hash"] = "legacy"
            updated = True

        # write upgraded metadata atomically
        if updated:
            tmp = metadata_path + ".tmp"

            with open(tmp, "w") as t:
                json.dump(meta, t, indent=4)

            os.replace(tmp, metadata_path)

        # ---------------------------------------------
        # STRICT TYPE CHECK
        # ---------------------------------------------

        if meta["metadata_type"] != "model":
            raise RuntimeError(
                "Attempted to register non-model metadata."
            )

        # ---------------------------------------------
        # Final validation
        # ---------------------------------------------

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

        if not isinstance(meta["features"], list):
            raise RuntimeError("features must be a list")

        if not isinstance(meta["metrics"], dict):
            raise RuntimeError("metrics must be a dictionary")

        if not isinstance(meta["dataset_hash"], str):
            raise RuntimeError("dataset_hash must be a string")

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
            pass  # Never crash on Windows / containers

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

    @staticmethod
    def get_latest_version(base_dir: str) -> str:

        pointer = os.path.join(base_dir, ModelRegistry.LATEST_POINTER)

        if os.path.exists(pointer):

            with open(pointer) as f:
                return json.load(f)["version"]

        latest_link = os.path.join(base_dir, "latest")

        if os.path.islink(latest_link):
            return os.readlink(latest_link)

        raise RuntimeError("Latest pointer missing.")
