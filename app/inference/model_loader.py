import os
import joblib
import logging
import threading
import json
import hashlib
from dataclasses import dataclass
from glob import glob

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES
)

from app.monitoring.metrics import MODEL_VERSION

logger = logging.getLogger("marketsentinel.loader")


############################################################
# MODEL CONTAINER
############################################################

@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    schema_signature: str
    dataset_hash: str
    training_code_hash: str
    artifact_hash: str


############################################################
# SINGLETON LOADER (XGBOOST ONLY)
############################################################

class ModelLoader:

    _instance = None
    _lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 50_000

    ########################################################

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    ########################################################

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

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
    # FIND LATEST MODEL ARTIFACT
    ########################################################

    def _find_latest_artifact(self, base_dir):

        model_files = glob(os.path.join(base_dir, "model_*.pkl"))

        if not model_files:
            raise RuntimeError("No trained model artifacts found.")

        latest = max(model_files, key=os.path.getmtime)

        timestamp = os.path.basename(latest).split("_")[1].split(".")[0]

        metadata_path = os.path.join(
            base_dir,
            f"metadata_{timestamp}.json"
        )

        if not os.path.exists(metadata_path):
            raise RuntimeError("Metadata file missing for latest model.")

        return latest, metadata_path, timestamp

    ########################################################
    # VALIDATE METADATA
    ########################################################

    def _validate_metadata(self, metadata_path):

        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("schema_signature") != get_schema_signature():
            raise RuntimeError("Schema mismatch during inference.")

        required_fields = [
            "dataset_hash",
            "training_code_hash",
            "schema_signature"
        ]

        for field in required_fields:
            if field not in meta:
                raise RuntimeError(f"Metadata missing required field: {field}")

        return meta

    ########################################################
    # SAFE MODEL LOAD
    ########################################################

    def _safe_load_model(self, model_path):

        if os.path.getsize(model_path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError("Artifact too small — corrupted model.")

        model = joblib.load(model_path)

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Invalid model artifact loaded.")

        return model

    ########################################################
    # RELOAD XGB IF NEEDED
    ########################################################

    def _reload_xgb_if_needed(self):

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

        meta = self._validate_metadata(metadata_path)

        artifact_hash = self._sha256(model_path)

        model = self._safe_load_model(model_path)

        container = LoadedModel(
            model=model,
            version=version,
            schema_signature=meta["schema_signature"],
            dataset_hash=meta["dataset_hash"],
            training_code_hash=meta["training_code_hash"],
            artifact_hash=artifact_hash
        )

        # Atomic assignment (important for thread safety)
        self._xgb_container = container

        MODEL_VERSION.labels(
            model="xgboost",
            version=version
        ).set(1)

        return container.model

    ########################################################
    # FEATURE VALIDATION
    ########################################################

    def validate_features(self, df):

        missing = set(MODEL_FEATURES) - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Inference feature mismatch: missing {missing}"
            )

        return df.loc[:, MODEL_FEATURES]

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
    # WARMUP (used in app.main)
    ########################################################

    def warmup(self):
        logger.info("Model warmup triggered.")
        _ = self.xgb