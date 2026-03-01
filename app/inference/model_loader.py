import os
import joblib
import logging
import threading
import hashlib
import json
from dataclasses import dataclass
from typing import Optional

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES,
    SCHEMA_VERSION,
)

from core.market.universe import MarketUniverse
from core.artifacts.metadata_manager import MetadataManager
from app.monitoring.metrics import MODEL_VERSION

logger = logging.getLogger("marketsentinel.loader")


# =========================================================
# LOADED MODEL CONTAINER
# =========================================================

@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    schema_signature: str
    dataset_hash: str
    training_code_hash: str
    artifact_hash: str
    feature_checksum: str
    pointer_hash: Optional[str]
    training_fingerprint: Optional[str]
    universe_hash: Optional[str]


# =========================================================
# MODEL LOADER (SINGLETON)
# =========================================================

class ModelLoader:

    _instance = None
    _instance_lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 50_000
    MIN_METADATA_BYTES = 500

    POINTER_FILENAME = "production_pointer.json"

    STRICT_GOVERNANCE = os.getenv("MODEL_STRICT_GOVERNANCE", "1") == "1"

    ########################################################

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    ########################################################

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._reload_lock = threading.Lock()
        self._xgb_container: Optional[LoadedModel] = None
        self._initialized = True

    ########################################################
    # HASH UTIL
    ########################################################

    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    ########################################################
    # FEATURE CHECKSUM
    ########################################################

    def _compute_feature_checksum(self, feature_list):
        canonical = json.dumps(
            list(feature_list),
            sort_keys=False
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    ########################################################
    # REGISTRY
    ########################################################

    def _get_registry_dir(self):

        base_dir = os.getenv(
            "XGB_REGISTRY_DIR",
            os.path.abspath("artifacts/xgboost")
        )

        base_dir = os.path.realpath(base_dir)

        if not os.path.isdir(base_dir):
            raise RuntimeError("Model registry directory missing.")

        return base_dir

    ########################################################
    # AUTO DISCOVERY (SAFE BOOTSTRAP)
    ########################################################

    def _auto_create_pointer_if_safe(self, base_dir):

        model_files = [
            f for f in os.listdir(base_dir)
            if f.startswith("model_") and f.endswith(".pkl")
        ]

        if len(model_files) != 1:
            raise RuntimeError(
                "production_pointer.json missing and multiple/no models found."
            )

        version = model_files[0].replace("model_", "").replace(".pkl", "")

        pointer_path = os.path.join(base_dir, self.POINTER_FILENAME)

        logger.warning(
            "Auto-creating production pointer for version=%s", version
        )

        with open(pointer_path, "w", encoding="utf-8") as f:
            json.dump({"model_version": version}, f)

        return pointer_path

    ########################################################
    # POINTER RESOLUTION
    ########################################################

    def _resolve_production_version(self, base_dir):

        pointer_path = os.path.join(base_dir, self.POINTER_FILENAME)

        if not os.path.exists(pointer_path):
            pointer_path = self._auto_create_pointer_if_safe(base_dir)

        if os.path.islink(pointer_path):
            raise RuntimeError("Symlinked production pointer detected.")

        if os.path.getsize(pointer_path) < 20:
            raise RuntimeError("Production pointer corrupted.")

        pointer_hash = self._sha256(pointer_path)

        with open(pointer_path, encoding="utf-8") as f:
            pointer = json.load(f)

        version = pointer.get("model_version")

        if not version:
            raise RuntimeError("Invalid production pointer format.")

        model_path = os.path.join(base_dir, f"model_{version}.pkl")
        metadata_path = os.path.join(base_dir, f"metadata_{version}.json")

        if not os.path.exists(model_path):
            raise RuntimeError("Production model missing.")

        if not os.path.exists(metadata_path):
            raise RuntimeError("Production metadata missing.")

        return model_path, metadata_path, version, pointer_hash

    ########################################################
    # SAFE MODEL LOAD
    ########################################################

    def _safe_load_model(self, model_path):

        if os.path.getsize(model_path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError("Artifact too small — corrupted.")

        model = joblib.load(model_path)

        if not hasattr(model, "predict"):
            raise RuntimeError("Invalid model artifact.")

        if not hasattr(model, "feature_names"):
            raise RuntimeError("Model missing feature_names contract.")

        if list(model.feature_names) != list(MODEL_FEATURES):
            raise RuntimeError("Model feature order mismatch.")

        return model

    ########################################################
    # RELOAD LOGIC
    ########################################################

    def _reload_xgb_if_needed(self):

        with self._reload_lock:

            base_dir = self._get_registry_dir()
            model_path, metadata_path, version, pointer_hash = \
                self._resolve_production_version(base_dir)

            if (
                self._xgb_container and
                self._xgb_container.version == version and
                self._xgb_container.pointer_hash == pointer_hash
            ):
                return self._xgb_container.model

            logger.info("Loading XGBoost version=%s", version)

            if os.path.getsize(metadata_path) < self.MIN_METADATA_BYTES:
                raise RuntimeError("Metadata file corrupted.")

            meta = MetadataManager.load_metadata(metadata_path)

            required_keys = {
                "metadata_type",
                "schema_signature",
                "schema_version",
                "features",
                "artifact_hash",
                "dataset_hash",
                "training_code_hash",
                "feature_checksum",
                "universe_hash",
            }

            missing = required_keys - set(meta.keys())
            if missing:
                raise RuntimeError(
                    f"Metadata missing required fields: {missing}"
                )

            if meta.get("metadata_type") != "training_manifest_v1":
                raise RuntimeError("Unsupported metadata type.")

            if meta.get("schema_signature") != get_schema_signature():
                raise RuntimeError("Schema signature mismatch.")

            if meta.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("Schema version drift detected.")

            if list(meta.get("features")) != list(MODEL_FEATURES):
                raise RuntimeError("Metadata feature mismatch.")

            artifact_hash_actual = self._sha256(model_path)

            if meta["artifact_hash"] != artifact_hash_actual:
                raise RuntimeError("Artifact tampering detected.")

            model = self._safe_load_model(model_path)

            feature_checksum_actual = \
                self._compute_feature_checksum(MODEL_FEATURES)

            if meta["feature_checksum"] != feature_checksum_actual:
                raise RuntimeError("Feature checksum mismatch.")

            universe_hash_current = MarketUniverse.fingerprint()

            if meta["universe_hash"] != universe_hash_current:
                raise RuntimeError("Universe fingerprint mismatch.")

            training_fingerprint = getattr(
                model,
                "training_fingerprint",
                None
            )

            new_container = LoadedModel(
                model=model,
                version=version,
                schema_signature=meta["schema_signature"],
                dataset_hash=meta["dataset_hash"],
                training_code_hash=meta["training_code_hash"],
                artifact_hash=artifact_hash_actual,
                feature_checksum=feature_checksum_actual,
                pointer_hash=pointer_hash,
                training_fingerprint=training_fingerprint,
                universe_hash=meta["universe_hash"]
            )

            self._xgb_container = new_container

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

            logger.info(
                "Model loaded | version=%s | artifact_hash=%s | universe_hash=%s",
                version,
                artifact_hash_actual[:12],
                meta["universe_hash"][:12]
            )

            return new_container.model

    ########################################################
    # PUBLIC ACCESSORS
    ########################################################

    @property
    def xgb(self):
        return self._reload_xgb_if_needed()

    @property
    def xgb_version(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.version

    @property
    def schema_signature(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.schema_signature

    @property
    def dataset_hash(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.dataset_hash

    @property
    def training_code_hash(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.training_code_hash

    @property
    def artifact_hash(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.artifact_hash

    @property
    def feature_checksum(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.feature_checksum

    ########################################################
    # FEATURE IMPORTANCE
    ########################################################

    def get_feature_importance(self):

        model = self.xgb

        if not hasattr(model, "export_feature_importance"):
            raise RuntimeError(
                "Model does not support feature importance export."
            )

        importance = model.export_feature_importance()

        return {
            "model_version": self.xgb_version,
            "feature_checksum": self.feature_checksum,
            "importance": importance
        }

    ########################################################
    # WARMUP
    ########################################################

    def warmup(self):
        logger.info("Model warmup triggered.")
        _ = self.xgb