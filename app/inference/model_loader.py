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
    artifact_hash: str
    feature_checksum: str
    pointer_hash: Optional[str]
    training_fingerprint: Optional[str]


# =========================================================
# MODEL LOADER (CV-FRIENDLY HARDENED)
# =========================================================

class ModelLoader:

    _instance = None
    _instance_lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 20_000
    MIN_METADATA_BYTES = 300
    POINTER_FILENAME = "production_pointer.json"

    STRICT_GOVERNANCE = os.getenv("MODEL_STRICT_GOVERNANCE", "0") == "1"

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

    def _compute_feature_checksum(self):
        canonical = json.dumps(list(MODEL_FEATURES), sort_keys=False).encode()
        return hashlib.sha256(canonical).hexdigest()

    ########################################################
    # REGISTRY
    ########################################################

    def _get_registry_dir(self):
        base_dir = os.getenv(
            "XGB_REGISTRY_DIR",
            os.path.abspath("artifacts/xgboost")
        )

        if not os.path.isdir(base_dir):
            raise RuntimeError("Model registry directory missing.")

        return base_dir

    ########################################################
    # POINTER RESOLUTION
    ########################################################

    def _resolve_production_version(self, base_dir):

        pointer_path = os.path.join(base_dir, self.POINTER_FILENAME)

        if not os.path.exists(pointer_path):
            raise RuntimeError("Production pointer missing.")

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

            # =====================================================
            # BASIC GOVERNANCE CHECKS (CV MODE)
            # =====================================================

            actual_hash = self._sha256(model_path)
            if meta.get("artifact_hash") != actual_hash:
                raise RuntimeError("Artifact tampering detected.")

            if meta.get("schema_signature") != get_schema_signature():
                logger.warning("Schema signature mismatch.")

            if meta.get("schema_version") != SCHEMA_VERSION:
                logger.warning("Schema version drift detected.")

            if list(meta.get("features")) != list(MODEL_FEATURES):
                logger.warning("Metadata feature mismatch.")

            if meta.get("feature_checksum") != self._compute_feature_checksum():
                logger.warning("Feature checksum drift detected.")

            try:
                universe_hash_current = MarketUniverse.fingerprint()
                if meta.get("universe_hash") != universe_hash_current:
                    logger.warning("Universe drift detected (soft).")
            except Exception:
                pass

            model = self._safe_load_model(model_path)

            training_fingerprint = getattr(
                model,
                "training_fingerprint",
                None
            )

            new_container = LoadedModel(
                model=model,
                version=version,
                schema_signature=meta.get("schema_signature"),
                dataset_hash=meta.get("dataset_hash"),
                artifact_hash=meta.get("artifact_hash"),
                feature_checksum=meta.get("feature_checksum"),
                pointer_hash=pointer_hash,
                training_fingerprint=training_fingerprint
            )

            self._xgb_container = new_container

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

            logger.info(
                "Model loaded | version=%s | artifact=%s",
                version,
                meta["artifact_hash"][:12]
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
    def feature_checksum(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.feature_checksum

    @property
    def dataset_hash(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.dataset_hash

    @property
    def training_fingerprint(self):
        self._reload_xgb_if_needed()
        return self._xgb_container.training_fingerprint

    ########################################################
    # FEATURE IMPORTANCE
    ########################################################

    def get_feature_importance(self):

        model = self.xgb

        if not hasattr(model, "export_feature_importance"):
            raise RuntimeError(
                "Model does not support feature importance export."
            )

        return model.export_feature_importance()

    ########################################################
    # WARMUP
    ########################################################

    def warmup(self):
        logger.info("Model warmup triggered.")
        _ = self.xgb