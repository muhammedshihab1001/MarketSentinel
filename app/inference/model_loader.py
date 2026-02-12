import os
import joblib
import logging
import threading
import json
from typing import Optional

from core.artifacts.model_registry import ModelRegistry
from core.schema.feature_schema import get_schema_signature
from models.lstm_model import forecast_lstm
from models.sarimax_model import SarimaxModel

from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


class ModelLoader:
    """
    Institutional production model loader.

    Guarantees:
    - registry-only loading
    - metadata validation
    - schema enforcement
    - wrapper verification
    - thread-safe lazy loading
    - fail-closed behavior
    """

    _instance = None
    _instance_lock = threading.Lock()

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
    # REGISTRY RESOLUTION
    ###################################################

    def _resolve_latest(self, base_dir: str):

        latest_path = os.path.join(
            base_dir,
            ModelRegistry.LATEST_POINTER
        )

        if not os.path.exists(latest_path):
            raise RuntimeError(
                f"Registry pointer missing: {latest_path}"
            )

        with open(latest_path) as f:
            payload = json.load(f)

        version = payload.get("version")

        if not version:
            raise RuntimeError("Registry pointer corrupted.")

        version_dir = os.path.join(base_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError(
                "Registry version directory missing."
            )

        return version, version_dir

    ###################################################
    # METADATA VALIDATION
    ###################################################

    def _validate_metadata(self, metadata_path: str):

        if not os.path.exists(metadata_path):
            raise RuntimeError("Model metadata missing.")

        with open(metadata_path) as f:
            meta = json.load(f)

        if meta.get("schema_signature") != get_schema_signature():
            raise RuntimeError(
                "Schema mismatch — model incompatible with runtime."
            )

        return meta

    ###################################################
    # SAFE LOAD
    ###################################################

    def _safe_joblib_load(self, path):

        if not os.path.exists(path):
            raise RuntimeError(f"Artifact missing: {path}")

        try:
            return joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"Artifact appears corrupted: {path}"
            ) from exc

    ###################################################
    # XGBOOST
    ###################################################

    def _reload_xgb_if_needed(self):

        base_dir = os.getenv(
            "XGB_REGISTRY_DIR",
            "artifacts/xgboost"
        )

        version, version_dir = self._resolve_latest(base_dir)

        if self._xgb is not None and self._xgb_version == version:
            return self._xgb

        with self._load_lock:

            if self._xgb is not None and self._xgb_version == version:
                return self._xgb

            logger.warning(
                f"Loading XGBoost model from registry version={version}"
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

        version, version_dir = self._resolve_latest(base_dir)

        if self._sarimax is not None and self._sarimax_version == version:
            return self._sarimax

        with self._load_lock:

            if self._sarimax is not None and self._sarimax_version == version:
                return self._sarimax

            logger.warning(
                f"Loading SARIMAX model from registry version={version}"
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
    # WARMUP
    ###################################################

    def warmup(self):

        if os.getenv("MODEL_WARMUP", "true") != "true":
            return

        logger.info("Warming models")

        _ = self.xgb
        _ = self.sarimax

        logger.info("Models ready")

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
