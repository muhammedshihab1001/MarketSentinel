import os
import joblib
import logging
import threading
import hashlib
import numpy as np
import json
from dataclasses import dataclass
from glob import glob

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES,
    SCHEMA_VERSION,
    DTYPE
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


############################################################
# SINGLETON LOADER
############################################################

class ModelLoader:

    _instance = None
    _instance_lock = threading.Lock()
    MIN_ARTIFACT_BYTES = 50_000

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

    ########################################################
    # SHA256
    ########################################################

    def _sha256(self, path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    ########################################################
    # FEATURE CHECKSUM
    ########################################################

    def _compute_feature_checksum(self, feature_list):
        canonical = json.dumps(feature_list, sort_keys=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    ########################################################
    # FIND LATEST MODEL
    ########################################################

    def _find_latest_artifact(self, base_dir):

        if not os.path.exists(base_dir):
            raise RuntimeError(f"Registry directory missing: {base_dir}")

        model_files = glob(os.path.join(base_dir, "model_*.pkl"))

        if not model_files:
            raise RuntimeError("No trained model artifacts found.")

        latest = max(model_files, key=os.path.getmtime)
        filename = os.path.basename(latest)

        if not filename.startswith("model_") or not filename.endswith(".pkl"):
            raise RuntimeError("Unexpected artifact naming format.")

        version = filename[len("model_"):-len(".pkl")]

        metadata_path = os.path.join(
            base_dir,
            f"metadata_{version}.json"
        )

        if not os.path.exists(metadata_path):
            raise RuntimeError("Metadata file missing for latest model.")

        return latest, metadata_path, version

    ########################################################
    # SAFE LOAD MODEL
    ########################################################

    def _safe_load_model(self, model_path):

        if os.path.getsize(model_path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError("Artifact too small — corrupted model.")

        model = joblib.load(model_path)

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Invalid model artifact loaded.")

        # STRICT FEATURE CONTRACT CHECK
        if hasattr(model, "feature_names"):
            trained_features = list(model.feature_names)
            if trained_features != MODEL_FEATURES:
                raise RuntimeError(
                    "Model feature contract mismatch at inference."
                )

        return model

    ########################################################
    # RELOAD IF NEEDED (ATOMIC SWAP)
    ########################################################

    def _reload_xgb_if_needed(self):

        with self._reload_lock:

            base_dir = os.getenv(
                "XGB_REGISTRY_DIR",
                os.path.abspath("artifacts/xgboost")
            )

            model_path, metadata_path, version = \
                self._find_latest_artifact(base_dir)

            if (
                self._xgb_container and
                self._xgb_container.version == version
            ):
                return self._xgb_container.model

            logger.info("Loading XGBoost version=%s", version)

            meta = MetadataManager.load_metadata(metadata_path)

            ####################################################
            # STRICT METADATA VALIDATION
            ####################################################

            required_fields = {
                "schema_signature",
                "schema_version",
                "features",
                "dataset_hash",
                "training_code_hash"
            }

            if not required_fields.issubset(meta.keys()):
                raise RuntimeError("Incomplete metadata manifest.")

            if meta.get("metadata_type") != "training_manifest_v1":
                raise RuntimeError("Unsupported metadata type.")

            if meta.get("schema_signature") != get_schema_signature():
                raise RuntimeError("Schema mismatch during inference.")

            if meta.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("Schema version drift detected.")

            if meta.get("features") != MODEL_FEATURES:
                raise RuntimeError("Metadata feature contract mismatch.")

            ####################################################
            # ARTIFACT INTEGRITY CHECK
            ####################################################

            artifact_hash_actual = self._sha256(model_path)

            if meta.get("artifact_hash"):
                if meta["artifact_hash"] != artifact_hash_actual:
                    raise RuntimeError("Artifact tampering detected.")

            ####################################################
            # SAFE LOAD
            ####################################################

            model = self._safe_load_model(model_path)

            ####################################################
            # FEATURE CHECKSUM (CRITICAL FIX)
            ####################################################

            feature_checksum = self._compute_feature_checksum(
                MODEL_FEATURES
            )

            new_container = LoadedModel(
                model=model,
                version=version,
                schema_signature=meta["schema_signature"],
                dataset_hash=meta["dataset_hash"],
                training_code_hash=meta["training_code_hash"],
                artifact_hash=artifact_hash_actual,
                feature_checksum=feature_checksum
            )

            self._xgb_container = new_container

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

            logger.info("Model loaded successfully | version=%s", version)

            return new_container.model

    ########################################################
    # STRICT FEATURE VALIDATION
    ########################################################

    def validate_features(self, df):

        missing = set(MODEL_FEATURES) - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Inference feature mismatch: missing {missing}"
            )

        df = df.loc[:, MODEL_FEATURES]

        if df.isnull().any().any():
            raise RuntimeError("NaN detected in inference features.")

        if not np.isfinite(df.values).all():
            raise RuntimeError("Non-finite values detected in inference.")

        return df.astype(DTYPE)

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

    ########################################################
    # 🔥 NEW: FEATURE CHECKSUM PROPERTY (FIX)
    ########################################################

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