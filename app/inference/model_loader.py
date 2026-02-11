import os
import joblib
import tensorflow as tf
import logging
import threading

from models.lstm_model import forecast_lstm
from models.prophet_model import forecast_prophet

from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


class ModelLoader:
    """
    Lightweight production model loader.

    Guarantees:
    - singleton instance
    - thread-safe lazy loading
    - optional hot reload
    - inference-safe validation
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

        self._xgb = None
        self._xgb_mtime = None

        self._lstm = None
        self._scaler = None
        self._prophet = None

        self._load_lock = threading.Lock()

        self._initialized = True

    # ---------------------------------------------------

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

    # ---------------------------------------------------
    # SIMPLE MODEL PATH
    # ---------------------------------------------------

    def _model_path(self):
        return os.getenv(
            "XGB_MODEL_PATH",
            "artifacts/xgboost/model.pkl"
        )

    # ---------------------------------------------------
    # HOT RELOAD VIA MTIME
    # ---------------------------------------------------

    def _reload_if_needed(self):

        path = self._model_path()

        if not os.path.exists(path):
            raise RuntimeError(f"Model not found at {path}")

        mtime = os.path.getmtime(path)

        if self._xgb is not None and self._xgb_mtime == mtime:
            return self._xgb

        with self._load_lock:

            if self._xgb is not None and self._xgb_mtime == mtime:
                return self._xgb

            logger.warning("Loading XGBoost model from disk")

            model = joblib.load(path)

            if not hasattr(model, "predict_proba"):
                raise RuntimeError(
                    "Loaded model missing predict_proba"
                )

            self._xgb = model
            self._xgb_mtime = mtime

            MODEL_VERSION.labels(
                model="xgboost",
                version=str(int(mtime))
            ).set(1)

        return self._xgb

    # ---------------------------------------------------
    # PUBLIC ACCESSOR
    # ---------------------------------------------------

    @property
    def xgb(self):
        return self._reload_if_needed()

    # ---------------------------------------------------
    # OPTIONAL WARMUP
    # ---------------------------------------------------

    def warmup(self):

        if os.getenv("MODEL_WARMUP", "true") != "true":
            return

        logger.info("Warming model")

        _ = self.xgb

        logger.info("Model ready")

    # ---------------------------------------------------
    # OPTIONAL FORECASTS
    # ---------------------------------------------------

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

    def prophet_forecast(self):

        if self._prophet is None:
            raise RuntimeError(
                "Prophet model not initialized."
            )

        return forecast_prophet(self._prophet)
