import os
import json
import datetime
import shutil
import hashlib
import uuid
import time
from typing import Dict, Any

from core.schema.feature_schema import get_schema_signature
from core.artifacts.metadata_manager import MetadataManager


class ModelRegistry:

    MANIFEST_NAME = "manifest.json"
    LATEST_POINTER = "latest.json"
    PROMOTION_LOCK = ".promotion.lock"

    LOCK_TIMEOUT_SECONDS = 600

    DEFAULT_MIN_BYTES = 10_000

    ########################################################
    # VERSION
    ########################################################

    @staticmethod
    def _version() -> str:
        ts = datetime.datetime.utcnow().strftime("v%Y_%m_%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{ts}_{suffix}"

    ########################################################
    # LOCKING
    ########################################################

    @staticmethod
    def _acquire_lock(base_dir: str):

        lock_path = os.path.join(base_dir, ModelRegistry.PROMOTION_LOCK)

        start = time.time()

        while True:

            try:
                fd = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR
                )

                os.write(fd, str(os.getpid()).encode())
                os.close(fd)

                return lock_path

            except FileExistsError:

                if time.time() - start > ModelRegistry.LOCK_TIMEOUT_SECONDS:
                    raise RuntimeError(
                        "Promotion lock timeout — possible dead registry."
                    )

                time.sleep(1.5)

    @staticmethod
    def _release_lock(lock_path: str):

        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except Exception:
                pass

    ########################################################
    # SAFETY
    ########################################################

    @staticmethod
    def _safe_join(base, *paths):

        base_real = os.path.realpath(base)
        target = os.path.realpath(os.path.join(base, *paths))

        if not target.startswith(base_real):
            raise RuntimeError("Path traversal detected in registry.")

        return target

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
    # METADATA HARD VALIDATION
    ########################################################

    @staticmethod
    def _validate_metadata_structure(meta: dict):

        required = [
            "dataset_hash",
            "schema_signature",
            "training_window",
            "training_universe",
            "features",
            "metrics"
        ]

        for r in required:
            if r not in meta:
                raise RuntimeError(f"Metadata missing required field: {r}")

        if meta["schema_signature"] != get_schema_signature():
            raise RuntimeError(
                "Metadata schema does not match runtime schema."
            )

        universe = meta["training_universe"]

        if not isinstance(universe, dict):
            raise RuntimeError("Invalid training_universe.")

        if "tickers" not in universe:
            raise RuntimeError("Universe missing tickers.")

    ########################################################
    # MANIFEST HASH
    ########################################################

    @staticmethod
    def _manifest_hash(manifest: dict):

        clone = dict(manifest)
        clone.pop("manifest_integrity_hash", None)

        canonical = json.dumps(
            clone,
            sort_keys=True,
            separators=(",", ":")
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

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
    # REGISTER MODEL
    ########################################################

    @staticmethod
    def register_model(
        base_dir: str,
        model_path: str,
        metadata_path: str,
        parent_version: str | None = None
    ) -> str:

        base_dir = os.path.realpath(base_dir)
        os.makedirs(base_dir, exist_ok=True)

        lock = ModelRegistry._acquire_lock(base_dir)

        try:

            meta = MetadataManager.load_metadata(metadata_path)
            ModelRegistry._validate_metadata_structure(meta)

            version = ModelRegistry._version()

            version_dir = ModelRegistry._safe_join(base_dir, version)
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
                    "training_window": meta["training_window"],

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

                return version

            except Exception:

                if os.path.exists(staging_dir):
                    shutil.rmtree(staging_dir, ignore_errors=True)

                raise

        finally:
            ModelRegistry._release_lock(lock)

    ########################################################
    # 🔥 PROMOTION (YOU WERE MISSING THIS)
    ########################################################

    @staticmethod
    def promote_to_production(base_dir: str, version: str):

        version_dir = ModelRegistry._safe_join(base_dir, version)

        manifest_path = ModelRegistry._safe_join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        if not os.path.exists(manifest_path):
            raise RuntimeError("Manifest missing — cannot promote.")

        with open(manifest_path) as f:
            manifest = json.load(f)

        manifest["stage"] = "production"

        manifest["history"].append({
            "event": "promoted",
            "utc": datetime.datetime.utcnow().isoformat()
        })

        manifest["manifest_integrity_hash"] = (
            ModelRegistry._manifest_hash(manifest)
        )

        ModelRegistry._atomic_json_write(
            manifest_path,
            manifest
        )

        ####################################################
        # UPDATE LATEST POINTER
        ####################################################

        latest_path = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        pointer = {
            "version": version,
            "updated_utc": datetime.datetime.utcnow().isoformat()
        }

        ModelRegistry._atomic_json_write(
            latest_path,
            pointer
        )
