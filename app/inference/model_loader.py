import os
import joblib
import logging
import threading
import json
import hashlib
from dataclasses import dataclass

from core.artifacts.model_registry import ModelRegistry
from core.artifacts.metadata_manager import MetadataManager
from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES
)

from models.lstm_model import forecast_lstm
from models.sarimax_model import SarimaxModel

from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    dataset_hash: str


class ModelLoader:

    _instance = None
    _instance_lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 50_000

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

        self._load_lock = threading.RLock()

        self._initialized = True

    ###################################################
    # HASH
    ###################################################

    def _sha256(self, path):

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    ###################################################
    # PATH SAFETY
    ###################################################

    def _resolve_registry_path(self, base_dir):

        base_dir = os.path.realpath(os.path.abspath(base_dir))

        if not os.path.isdir(base_dir):
            raise RuntimeError(f"Invalid registry path: {base_dir}")

        if os.path.islink(base_dir):
            raise RuntimeError("Registry path cannot be a symlink.")

        return base_dir

    ###################################################
    # MANIFEST CHECK
    ###################################################

    def _verify_manifest_lineage(self, manifest):

        if manifest.get("schema_signature") != get_schema_signature():
            raise RuntimeError("Manifest schema mismatch.")

        required = ["dataset_hash", "training_code_hash"]

        for r in required:
            if r not in manifest:
                raise RuntimeError(f"Manifest missing {r}")

    ###################################################
    # REGISTRY RESOLUTION (FIXED)
    ###################################################

    def _resolve_latest_verified(self, base_dir: str):

        base_dir = self._resolve_registry_path(base_dir)

        version = ModelRegistry.load_latest_version(base_dir)

        ModelRegistry.verify_artifacts(base_dir, version)

        version_dir = os.path.realpath(os.path.join(base_dir, version))

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        with open(manifest_path) as f:
            manifest = json.load(f)

        if manifest.get("stage") != "production":
            raise RuntimeError("Refusing to load non-production model.")

        self._verify_manifest_lineage(manifest)

        return version, version_dir, manifest

    ###################################################
    # METADATA VALIDATION (MODEL-AWARE)
    ###################################################

    def _validate_metadata(self, metadata_path: str, manifest: dict, model_type: str):

        meta = MetadataManager.load_metadata(metadata_path)

        if meta.get("schema_signature") != get_schema_signature():
            raise RuntimeError("Schema mismatch.")

        # Only enforce feature contract for XGBoost
        if model_type == "xgboost":
            if meta.get("features") != list(MODEL_FEATURES):
                raise RuntimeError("Feature mismatch.")

        if meta.get("dataset_hash") != manifest.get("dataset_hash"):
            raise RuntimeError("Dataset lineage mismatch.")

        if meta.get("training_code_hash") != manifest.get("training_code_hash"):
            raise RuntimeError("Training code lineage mismatch.")

        return meta

    ###################################################
    # SAFE LOAD
    ###################################################

    def _safe_joblib_load(self, path, expected_hash):

        if os.path.getsize(path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError("Artifact too small.")

        actual = self._sha256(path)

        if actual != expected_hash:
            raise RuntimeError("Artifact hash mismatch vs manifest.")

        model = joblib.load(path)

        return model

    ###################################################
    # XGBOOST
    ###################################################

    def _reload_xgb_if_needed(self):

        with self._load_lock:

            base_dir = os.getenv(
                "XGB_REGISTRY_DIR",
                os.path.abspath("artifacts/xgboost")
            )

            version, version_dir, manifest = \
                self._resolve_latest_verified(base_dir)

            if self._xgb_container and self._xgb_container.version == version:
                return self._xgb_container.model

            logger.info("Loading XGBoost version=%s", version)

            model_path = os.path.join(version_dir, "model.pkl")
            metadata_path = os.path.join(version_dir, "metadata.json")

            meta = self._validate_metadata(
                metadata_path,
                manifest,
                "xgboost"
            )

            expected_hash = manifest["artifacts"]["model.pkl"]

            model = self._safe_joblib_load(
                model_path,
                expected_hash
            )

            if not hasattr(model, "predict_proba"):
                raise RuntimeError("Invalid XGBoost artifact.")

            new_container = LoadedModel(
                model=model,
                version=version,
                dataset_hash=meta["dataset_hash"]
            )

            self._xgb_container = new_container

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

            return model

    ###################################################
    # SARIMAX
    ###################################################

    def _reload_sarimax_if_needed(self):

        with self._load_lock:

            base_dir = os.getenv(
                "SARIMAX_REGISTRY_DIR",
                os.path.abspath("artifacts/sarimax")
            )

            version, version_dir, manifest = \
                self._resolve_latest_verified(base_dir)

            if self._sarimax_container and self._sarimax_container.version == version:
                return self._sarimax_container.model

            logger.info("Loading SARIMAX version=%s", version)

            model_path = os.path.join(version_dir, "model.pkl")
            metadata_path = os.path.join(version_dir, "metadata.json")

            meta = self._validate_metadata(
                metadata_path,
                manifest,
                "sarimax"
            )

            expected_hash = manifest["artifacts"]["model.pkl"]

            model = self._safe_joblib_load(
                model_path,
                expected_hash
            )

            if not isinstance(model, SarimaxModel):
                raise RuntimeError("Invalid SARIMAX artifact.")

            self._sarimax_container = LoadedModel(
                model=model,
                version=version,
                dataset_hash=meta["dataset_hash"]
            )

            MODEL_VERSION.labels(
                model="sarimax",
                version=version
            ).set(1)

            return model

    ###################################################
    # LSTM LOADER (NEW — CRITICAL)
    ###################################################

    def _load_lstm_if_needed(self):

        if self._lstm is not None:
            return

        base_dir = os.getenv(
            "LSTM_REGISTRY_DIR",
            os.path.abspath("artifacts/lstm")
        )

        version, version_dir, manifest = \
            self._resolve_latest_verified(base_dir)

        from tensorflow.keras.models import load_model

        model_path = os.path.join(version_dir, "model.keras")
        scaler_path = os.path.join(version_dir, "scalers.pkl")

        self._lstm = load_model(model_path)
        self._scaler = joblib.load(scaler_path)

        MODEL_VERSION.labels(
            model="lstm",
            version=version
        ).set(1)

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
    # WARMUP
    ###################################################

    def warmup(self):

        if os.getenv("MODEL_WARMUP", "true").lower() != "true":
            return

        logger.info("Warming models")

        _ = self.xgb
        _ = self.sarimax
        self._load_lstm_if_needed()

        logger.info("Model warmup complete")

    ###################################################
    # LSTM FORECAST
    ###################################################

    def lstm_forecast(self, recent_prices):

        self._load_lstm_if_needed()

        return forecast_lstm(
            self._lstm,
            self._scaler,
            recent_prices
        )
