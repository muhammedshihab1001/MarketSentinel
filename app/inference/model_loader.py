import os
import joblib
import logging
import threading
import hashlib
import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES,
    SCHEMA_VERSION,
)

from core.artifacts.metadata_manager import MetadataManager
from app.monitoring.metrics import MODEL_VERSION

logger = logging.getLogger("marketsentinel.loader")


############################################################
# MODEL CONTAINER (IMMUTABLE)
############################################################

@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    schema_signature: str
    dataset_hash: str
    training_code_hash: str
    artifact_hash: str
    feature_checksum: str
    pointer_hash: str | None


############################################################
# SINGLETON LOADER
############################################################

class ModelLoader:

    _instance = None
    _instance_lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 50_000
    POINTER_FILENAME = "production_pointer.json"

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
        self._xgb_container: LoadedModel | None = None
        self._initialized = True

        self._allow_latest_fallback = (
            os.getenv("ALLOW_LATEST_FALLBACK", "false").lower() == "true"
        )

    ########################################################
    # SHA256
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
    # REGISTRY DIR VALIDATION
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
    # PRODUCTION POINTER RESOLUTION
    ########################################################

    def _resolve_production_version(self, base_dir):

        pointer_path = os.path.join(
            base_dir,
            self.POINTER_FILENAME
        )

        if not os.path.exists(pointer_path):

            if self._allow_latest_fallback:
                logger.warning("Pointer missing — fallback enabled.")
                return None

            raise RuntimeError("production_pointer.json missing.")

        if os.path.islink(pointer_path):
            raise RuntimeError("Symlinked production pointer detected.")

        pointer_hash = self._sha256(pointer_path)

        with open(pointer_path, encoding="utf-8") as f:
            pointer = json.load(f)

        version = pointer.get("model_version")

        if not version:
            raise RuntimeError("Invalid production pointer format.")

        model_path = os.path.join(
            base_dir,
            f"model_{version}.pkl"
        )

        metadata_path = os.path.join(
            base_dir,
            f"metadata_{version}.json"
        )

        if not os.path.exists(model_path):
            raise RuntimeError("Production pointer model missing.")

        if not os.path.exists(metadata_path):
            raise RuntimeError("Production pointer metadata missing.")

        return model_path, metadata_path, version, pointer_hash

    ########################################################
    # FIND ARTIFACT
    ########################################################

    def _find_artifact(self):

        base_dir = self._get_registry_dir()

        resolved = self._resolve_production_version(base_dir)

        if resolved:
            return resolved

        model_files = glob(
            os.path.join(base_dir, "model_*.pkl")
        )

        if not model_files:
            raise RuntimeError("No model artifacts found.")

        latest = max(model_files, key=os.path.getmtime)
        version = Path(latest).stem.replace("model_", "")

        metadata_path = os.path.join(
            base_dir,
            f"metadata_{version}.json"
        )

        if not os.path.exists(metadata_path):
            raise RuntimeError("Metadata missing for latest model.")

        return latest, metadata_path, version, None

    ########################################################
    # SAFE LOAD MODEL
    ########################################################

    def _safe_load_model(self, model_path):

        if os.path.getsize(model_path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError("Artifact too small — corrupted.")

        model = joblib.load(model_path)

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Invalid model artifact.")

        if hasattr(model, "feature_names"):
            if list(model.feature_names) != list(MODEL_FEATURES):
                raise RuntimeError("Model feature contract mismatch.")

        return model

    ########################################################
    # RELOAD IF NEEDED
    ########################################################

    def _reload_xgb_if_needed(self):

        with self._reload_lock:

            model_path, metadata_path, version, pointer_hash = \
                self._find_artifact()

            if (
                self._xgb_container and
                self._xgb_container.version == version and
                self._xgb_container.pointer_hash == pointer_hash
            ):
                return self._xgb_container.model

            logger.info("Loading XGBoost version=%s", version)

            meta = MetadataManager.load_metadata(metadata_path)

            ####################################################
            # STRICT METADATA VALIDATION
            ####################################################

            required_keys = {
                "metadata_type",
                "schema_signature",
                "schema_version",
                "features",
                "artifact_hash",
                "dataset_hash",
                "training_code_hash"
            }

            if not required_keys.issubset(meta.keys()):
                raise RuntimeError("Metadata missing required fields.")

            if meta.get("metadata_type") != "training_manifest_v1":
                raise RuntimeError("Unsupported metadata type.")

            if meta.get("schema_signature") != get_schema_signature():
                raise RuntimeError("Schema signature mismatch.")

            if meta.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("Schema version drift detected.")

            if list(meta.get("features")) != list(MODEL_FEATURES):
                raise RuntimeError("Metadata feature mismatch.")

            ####################################################
            # ARTIFACT INTEGRITY
            ####################################################

            artifact_hash_actual = self._sha256(model_path)

            if meta["artifact_hash"] != artifact_hash_actual:
                raise RuntimeError("Artifact tampering detected.")

            ####################################################
            # SAFE LOAD
            ####################################################

            model = self._safe_load_model(model_path)

            ####################################################
            # FEATURE CHECKSUM VALIDATION
            ####################################################

            feature_checksum_actual = \
                self._compute_feature_checksum(MODEL_FEATURES)

            if meta.get("feature_checksum") and \
                    meta["feature_checksum"] != feature_checksum_actual:
                raise RuntimeError("Feature checksum mismatch.")

            new_container = LoadedModel(
                model=model,
                version=version,
                schema_signature=meta["schema_signature"],
                dataset_hash=meta["dataset_hash"],
                training_code_hash=meta["training_code_hash"],
                artifact_hash=artifact_hash_actual,
                feature_checksum=feature_checksum_actual,
                pointer_hash=pointer_hash
            )

            self._xgb_container = new_container

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

            logger.info("Model loaded successfully | version=%s", version)

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
    # WARMUP
    ########################################################

    def warmup(self):
        logger.info("Model warmup triggered.")
        _ = self.xgb