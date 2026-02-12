import os
import joblib
import logging
import threading
import json
from dataclasses import dataclass

from core.artifacts.model_registry import ModelRegistry
from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES
)

from models.lstm_model import forecast_lstm
from models.sarimax_model import SarimaxModel

from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


###################################################
# IMMUTABLE CONTAINER
###################################################

@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    dataset_hash: str


class ModelLoader:

    _instance = None
    _instance_lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 50_000  # prevents zero-byte / truncated loads

    ###################################################

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    ###################################################

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        self._xgb_container: LoadedModel | None = None
        self._sarimax_container: LoadedModel | None = None

        self._lstm = None
        self._scaler = None

        self._load_lock = threading.Lock()

        self._initialized = True

    ###################################################
    # REGISTRY PATH SAFETY
    ###################################################

    def _validate_registry_path(self, base_dir):

        if not os.path.exists(base_dir):
            raise RuntimeError(
                f"Registry path does not exist: {base_dir}"
            )

    ###################################################
    # REGISTRY RESOLUTION
    ###################################################

    def _resolve_latest_verified(self, base_dir: str):

        self._validate_registry_path(base_dir)

        version = ModelRegistry.get_latest_version(base_dir)

        ModelRegistry.verify_artifacts(base_dir, version)

        version_dir = os.path.join(base_dir, version)

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        with open(manifest_path) as f:
            manifest = json.load(f)

        return version, version_dir, manifest

    ###################################################
    # METADATA CROSS VALIDATION
    ###################################################

    def _validate_metadata(self, metadata_path: str, manifest: dict):

        with open(metadata_path) as f:
            meta = json.load(f)

        if meta.get("schema_signature") != get_schema_signature():
            raise RuntimeError(
                "Schema mismatch — model incompatible with runtime."
            )

        if meta.get("features") != list(MODEL_FEATURES):
            raise RuntimeError(
                "Feature mismatch — runtime and model differ."
            )

        if meta.get("dataset_hash") != manifest.get("dataset_hash"):
            raise RuntimeError(
                "Dataset lineage mismatch between metadata and manifest."
            )

        if "training_code_hash" not in meta:
            raise RuntimeError("Training lineage missing.")

        return meta

    ###################################################
    # SAFE LOAD
    ###################################################

    def _safe_joblib_load(self, path):

        if not os.path.exists(path):
            raise RuntimeError(f"Artifact missing: {path}")

        if os.path.getsize(path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError(
                f"Artifact too small — likely corrupted: {path}"
            )

        try:
            model = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"Artifact appears corrupted: {path}"
            ) from exc

        return model

    ###################################################
    # XGBOOST
    ###################################################

    def _reload_xgb_if_needed(self):

        base_dir = os.getenv(
            "XGB_REGISTRY_DIR",
            "artifacts/xgboost"
        )

        version, version_dir, manifest = \
            self._resolve_latest_verified(base_dir)

        container = self._xgb_container

        if container and container.version == version:
            return container.model

        with self._load_lock:

            container = self._xgb_container

            if container and container.version == version:
                return container.model

            logger.warning(
                "Loading XGBoost model version=%s",
                version
            )

            model_path = os.path.join(version_dir, "model.pkl")
            metadata_path = os.path.join(version_dir, "metadata.json")

            meta = self._validate_metadata(metadata_path, manifest)

            model = self._safe_joblib_load(model_path)

            if not hasattr(model, "predict_proba"):
                raise RuntimeError(
                    "Loaded artifact missing predict_proba"
                )

            new_container = LoadedModel(
                model=model,
                version=version,
                dataset_hash=meta["dataset_hash"]
            )

            # ATOMIC SWAP
            self._xgb_container = new_container

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

        return self._xgb_container.model

    ###################################################
    # SARIMAX
    ###################################################

    def _reload_sarimax_if_needed(self):

        base_dir = os.getenv(
            "SARIMAX_REGISTRY_DIR",
            "artifacts/sarimax"
        )

        version, version_dir, manifest = \
            self._resolve_latest_verified(base_dir)

        container = self._sarimax_container

        if container and container.version == version:
            return container.model

        with self._load_lock:

            container = self._sarimax_container

            if container and container.version == version:
                return container.model

            logger.warning(
                "Loading SARIMAX model version=%s",
                version
            )

            model_path = os.path.join(version_dir, "model.pkl")
            metadata_path = os.path.join(version_dir, "metadata.json")

            meta = self._validate_metadata(metadata_path, manifest)

            model = self._safe_joblib_load(model_path)

            if not isinstance(model, SarimaxModel):
                raise RuntimeError(
                    "Loaded artifact is not a SarimaxModel wrapper."
                )

            _ = model.fitted_model

            new_container = LoadedModel(
                model=model,
                version=version,
                dataset_hash=meta["dataset_hash"]
            )

            self._sarimax_container = new_container

            MODEL_VERSION.labels(
                model="sarimax",
                version=version
            ).set(1)

        return self._sarimax_container.model

    ###################################################
    # PUBLIC ACCESSORS
    ###################################################

    @property
    def xgb(self):
        return self._reload_xgb_if_needed()

    @property
    def sarimax(self):
        return self._reload_sarimax_if_needed()

    ###################################################
    # FAIL-CLOSED WARMUP
    ###################################################

    def warmup(self):

        if os.getenv("MODEL_WARMUP", "true").lower() != "true":
            return

        logger.info("Warming models")

        _ = self.xgb
        _ = self.sarimax

        logger.info("Model warmup complete")

    ###################################################
    # LSTM
    ###################################################

    def lstm_forecast(self, recent_prices):

        if self._lstm is None or self._scaler is None:
            raise RuntimeError(
                "LSTM model not initialized."
            )

        return forecast_lstm(
            self._lstm,
            self._scaler,
            recent_prices
        )
