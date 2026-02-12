import os
import joblib
import tensorflow as tf
import logging
import threading
from typing import Optional

from models.lstm_model import forecast_lstm
from models.sarimax_model import SarimaxModel

from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


class ModelLoader:
    """
    Institutional production model loader.

    Guarantees:
    - singleton instance
    - thread-safe lazy loading
    - artifact validation
    - wrapper enforcement
    - hot reload
    - registry-compatible behavior
    """

    _instance = None
    _instance_lock = threading.Lock()
    _tf_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        self._configure_tensorflow()

        self._xgb: Optional[object] = None
        self._xgb_mtime: Optional[float] = None

        self._sarimax: Optional[SarimaxModel] = None
        self._sarimax_mtime: Optional[float] = None

        self._lstm = None
        self._scaler = None

        self._load_lock = threading.Lock()

        self._initialized = True

    ###################################################
    # TF CONFIG
    ###################################################

    def _configure_tensorflow(self):

        with self._tf_lock:

            try:

                disable_gpu = os.getenv(
                    "DISABLE_GPU",
                    "false"
                ).lower() == "true"

                if disable_gpu:
                    tf.config.set_visible_devices([], "GPU")

                intra = int(os.getenv("TF_INTRA_THREADS", "1"))
                inter = int(os.getenv("TF_INTER_THREADS", "1"))

                tf.config.threading.set_intra_op_parallelism_threads(intra)
                tf.config.threading.set_inter_op_parallelism_threads(inter)

            except Exception:
                logger.warning(
                    "TensorFlow already initialized — skipping device config."
                )

            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    ###################################################
    # PATH RESOLUTION
    ###################################################

    def _xgb_model_path(self):
        return os.getenv(
            "XGB_MODEL_PATH",
            "artifacts/xgboost/model.pkl"
        )

    def _sarimax_model_path(self):
        return os.getenv(
            "SARIMAX_MODEL_PATH",
            "artifacts/sarimax/model.pkl"
        )

    ###################################################
    # SAFE LOAD
    ###################################################

    def _safe_joblib_load(self, path):

        if not os.path.exists(path):
            raise RuntimeError(f"Model not found at {path}")

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

        path = self._xgb_model_path()
        mtime = os.path.getmtime(path)

        if self._xgb is not None and self._xgb_mtime == mtime:
            return self._xgb

        with self._load_lock:

            if self._xgb is not None and self._xgb_mtime == mtime:
                return self._xgb

            logger.warning("Loading XGBoost model from disk")

            model = self._safe_joblib_load(path)

            if not hasattr(model, "predict_proba"):
                raise RuntimeError(
                    "Loaded XGBoost artifact missing predict_proba"
                )

            self._xgb = model
            self._xgb_mtime = mtime

            MODEL_VERSION.labels(
                model="xgboost",
                version=str(int(mtime))
            ).set(1)

        return self._xgb

    ###################################################
    # SARIMAX
    ###################################################

    def _reload_sarimax_if_needed(self):

        path = self._sarimax_model_path()
        mtime = os.path.getmtime(path)

        if self._sarimax is not None and self._sarimax_mtime == mtime:
            return self._sarimax

        with self._load_lock:

            if self._sarimax is not None and self._sarimax_mtime == mtime:
                return self._sarimax

            logger.warning("Loading SARIMAX model from disk")

            model = self._safe_joblib_load(path)

            if not isinstance(model, SarimaxModel):
                raise RuntimeError(
                    "Loaded artifact is not a SarimaxModel wrapper."
                )

            # verify fitted state
            _ = model.fitted_model

            self._sarimax = model
            self._sarimax_mtime = mtime

            MODEL_VERSION.labels(
                model="sarimax",
                version=str(int(mtime))
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
    # LSTM FORECAST
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
