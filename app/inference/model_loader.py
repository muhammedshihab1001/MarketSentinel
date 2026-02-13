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

        base_dir = os.path.realpath(
            os.path.abspath(base_dir)
        )

        if not os.path.exists(base_dir):
            raise RuntimeError(
                f"Registry path does not exist: {base_dir}"
            )

        if not os.path.isdir(base_dir):
            raise RuntimeError("Registry path is not a directory.")

        if os.path.islink(base_dir):
            raise RuntimeError("Registry path cannot be a symlink.")

        return base_dir

    ###################################################
    # MANIFEST CROSS-CHECK (NEW — CRITICAL)
    ###################################################

    def _verify_manifest_lineage(self, manifest):

        if manifest.get("schema_signature") != get_schema_signature():
            raise RuntimeError(
                "Manifest schema mismatch — refusing load."
            )

        if "dataset_hash" not in manifest:
            raise RuntimeError("Manifest missing dataset hash.")

        if "training_code_hash" not in manifest:
            raise RuntimeError("Manifest missing code lineage.")

    ###################################################
    # REGISTRY RESOLUTION
    ###################################################

    def _resolve_latest_verified(self, base_dir: str):

        base_dir = self._resolve_registry_path(base_dir)

        version = ModelRegistry.get_latest_version(base_dir)

        if not version:
            raise RuntimeError(
                "No production model found in registry."
            )

        ModelRegistry.verify_artifacts(base_dir, version)

        version_dir = os.path.realpath(
            os.path.join(base_dir, version)
        )

        if not version_dir.startswith(base_dir):
            raise RuntimeError("Registry path traversal detected.")

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        with open(manifest_path) as f:
            manifest = json.load(f)

        if manifest.get("stage") != "production":
            raise RuntimeError(
                f"Refusing to load non-production model: {version}"
            )

        self._verify_manifest_lineage(manifest)

        return version, version_dir, manifest

    ###################################################
    # METADATA CROSS VALIDATION
    ###################################################

    def _validate_metadata(self, metadata_path: str, manifest: dict):

        meta = MetadataManager.load_metadata(metadata_path)

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

        if meta.get("training_code_hash") != manifest.get("training_code_hash"):
            raise RuntimeError(
                "Training code lineage mismatch."
            )

        if manifest.get("schema_signature") != meta.get("schema_signature"):
            raise RuntimeError(
                "Manifest and metadata schema mismatch."
            )

        if "training_universe" not in meta:
            raise RuntimeError(
                "Metadata missing training_universe."
            )

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

        pre_hash = self._sha256(path)

        try:
            model = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"Artifact appears corrupted: {path}"
            ) from exc

        post_hash = self._sha256(path)

        if pre_hash != post_hash:
            raise RuntimeError(
                "Artifact changed during load — disk instability suspected."
            )

        return model

    ###################################################
    # XGBOOST
    ###################################################

    def _reload_xgb_if_needed(self):

        base_dir = os.getenv(
            "XGB_REGISTRY_DIR",
            os.path.abspath("artifacts/xgboost")
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

            logger.info(
                "Loading XGBoost model version=%s",
                version
            )

            model_path = os.path.join(version_dir, "model.pkl")
            metadata_path = os.path.join(version_dir, "metadata.json")

            meta = self._validate_metadata(metadata_path, manifest)

            model = self._safe_joblib_load(model_path)

            ###################################################
            # HARD MODEL SANITY
            ###################################################

            if not hasattr(model, "predict_proba"):
                raise RuntimeError(
                    "Loaded artifact missing predict_proba"
                )

            if getattr(model, "n_features_in_", None) != len(MODEL_FEATURES):
                raise RuntimeError(
                    "Model feature count mismatch."
                )

            booster = model.get_booster()

            if booster.num_boosted_rounds() < 10:
                raise RuntimeError(
                    "Booster appears untrained."
                )

            import numpy as np

            sample = np.zeros((4, len(MODEL_FEATURES)))
            preds = model.predict_proba(sample)

            if preds.shape[1] != 2:
                raise RuntimeError(
                    "predict_proba must output 2-class probabilities."
                )

            if not np.isfinite(preds).all():
                raise RuntimeError(
                    "Model produced non-finite probabilities."
                )

            if np.std(preds) < 1e-6:
                raise RuntimeError(
                    "Model probabilities collapsed."
                )

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

        logger.info(
            "Loaded XGBoost | version=%s | dataset=%s",
            version,
            meta["dataset_hash"][:10]
        )

        return self._xgb_container.model

    ###################################################
    # VERSION ACCESSOR (CRITICAL FOR PIPELINE)
    ###################################################

    @property
    def xgb_version(self):
        _ = self.xgb
        return self._xgb_container.version

    ###################################################
    # SARIMAX
    ###################################################

    def _reload_sarimax_if_needed(self):

        base_dir = os.getenv(
            "SARIMAX_REGISTRY_DIR",
            os.path.abspath("artifacts/sarimax")
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

            logger.info(
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

        try:
            _ = self.xgb
            _ = self.sarimax
        except Exception:
            logger.critical("Model warmup failed — refusing to boot.")
            raise

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
