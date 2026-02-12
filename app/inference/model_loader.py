import os
import joblib
import logging
import threading
import json

from core.artifacts.model_registry import ModelRegistry
from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES
)

from models.lstm_model import forecast_lstm
from models.sarimax_model import SarimaxModel

from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


class ModelLoader:
    """
    Institutional production model loader.

    Guarantees:
    - registry integrity verification
    - schema enforcement
    - metadata enforcement
    - artifact hashing validation
    - thread-safe reload
    - atomic version swap
    - fail-closed behavior
    """

    _instance = None
    _instance_lock = threading.Lock()

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

        self._xgb = None
        self._xgb_version = None

        self._sarimax = None
        self._sarimax_version = None

        self._lstm = None
        self._scaler = None

        self._load_lock = threading.Lock()

        self._initialized = True

    ###################################################
    # REGISTRY RESOLUTION + VERIFICATION
    ###################################################

    def _resolve_latest_verified(self, base_dir: str):

        version = ModelRegistry.get_latest_version(base_dir)

        ModelRegistry.verify_artifacts(base_dir, version)

        version_dir = os.path.join(base_dir, version)

        return version, version_dir

    ###################################################
    # METADATA VALIDATION (STRICT)
    ###################################################

    def _validate_metadata(self, metadata_path: str):

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

        if "dataset_hash" not in meta:
            raise RuntimeError(
                "Metadata corrupted — dataset hash missing."
            )

        if "training_code_hash" not in meta:
            raise RuntimeError(
                "Metadata corrupted — lineage missing."
            )

        return meta

    ###################################################
    # SAFE LOAD
    ###################################################

    def _safe_joblib_load(self, path):

        if not os.path.exists(path):
            raise RuntimeError(f"Artifact missing: {path}")

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

        version, version_dir = self._resolve_latest_verified(base_dir)

        if self._xgb is not None and self._xgb_version == version:
            return self._xgb

        with self._load_lock:

            if self._xgb is not None and self._xgb_version == version:
                return self._xgb

            logger.warning(
                "Loading XGBoost model version=%s",
                version
            )

            model_path = os.path.join(version_dir, "model.pkl")
            metadata_path = os.path.join(version_dir, "metadata.json")

            self._validate_metadata(metadata_path)

            model = self._safe_joblib_load(model_path)

            if not hasattr(model, "predict_proba"):
                raise RuntimeError(
                    "Loaded artifact missing predict_proba"
                )

            self._xgb = model
            self._xgb_version = version

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

        return self._xgb

    ###################################################
    # SARIMAX
    ###################################################

    def _reload_sarimax_if_needed(self):

        base_dir = os.getenv(
            "SARIMAX_REGISTRY_DIR",
            "artifacts/sarimax"
        )

        version, version_dir = self._resolve_latest_verified(base_dir)

        if self._sarimax is not None and self._sarimax_version == version:
            return self._sarimax

        with self._load_lock:

            if self._sarimax is not None and self._sarimax_version == version:
                return self._sarimax

            logger.warning(
                "Loading SARIMAX model version=%s",
                version
            )

            model_path = os.path.join(version_dir, "model.pkl")
            metadata_path = os.path.join(version_dir, "metadata.json")

            self._validate_metadata(metadata_path)

            model = self._safe_joblib_load(model_path)

            if not isinstance(model, SarimaxModel):
                raise RuntimeError(
                    "Loaded artifact is not a SarimaxModel wrapper."
                )

            _ = model.fitted_model

            self._sarimax = model
            self._sarimax_version = version

            MODEL_VERSION.labels(
                model="sarimax",
                version=version
            ).set(1)

        return self._sarimax

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
    # SAFE WARMUP
    ###################################################

    def warmup(self):

        if os.getenv("MODEL_WARMUP", "true").lower() != "true":
            return

        logger.info("Warming models")

        try:
            _ = self.xgb
        except Exception:
            logger.exception("XGBoost warmup failed")

        try:
            _ = self.sarimax
        except Exception:
            logger.exception("SARIMAX warmup failed")

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
