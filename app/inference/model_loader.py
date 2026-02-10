import os
import json
import joblib
import tensorflow as tf
import logging
import threading

from models.lstm_model import forecast_lstm
from models.prophet_model import forecast_prophet
from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Institutional Registry-Aware Model Loader.

    Guarantees:
    ✅ Pointer-based resolution (cross-platform)
    ✅ Thread-safe lazy loading
    ✅ Metadata validation
    ✅ Cold-start prevention
    ✅ Singleton per worker
    ✅ TensorFlow memory safety
    """

    _instance = None
    _lock = threading.Lock()

    POINTER_FILE = "LATEST.json"

    # ---------------------------------------------------

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ---------------------------------------------------

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        # 🔥 TensorFlow CPU Safety
        tf.config.set_visible_devices([], "GPU")

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        self._xgb = None
        self._lstm = None
        self._scaler = None
        self._prophet = None

        self._initialized = True

    # ---------------------------------------------------
    # POINTER RESOLUTION
    # ---------------------------------------------------

    def _resolve_latest_dir(self, model_dir: str):

        pointer = os.path.join(model_dir, self.POINTER_FILE)

        if not os.path.exists(pointer):
            raise RuntimeError(
                f"No registry pointer found in {model_dir}. "
                f"Did training run?"
            )

        with open(pointer) as f:
            version = json.load(f)["latest"]

        version_dir = os.path.join(model_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError(
                f"Pointer references missing version: {version}"
            )

        return version_dir, version

    # ---------------------------------------------------
    # METADATA VALIDATION
    # ---------------------------------------------------

    def _validate_metadata(self, version_dir):

        meta_path = os.path.join(version_dir, "metadata.json")

        if not os.path.exists(meta_path):
            raise RuntimeError("Missing metadata.json")

        with open(meta_path) as f:
            meta = json.load(f)

        schema = meta.get("schema_version")

        if schema is None:
            raise RuntimeError("Metadata missing schema_version")

        # future-proof gate
        if float(schema.split(".")[0]) < 2:
            raise RuntimeError(
                "Outdated model schema. Retrain required."
            )

    # ---------------------------------------------------

    def _latest_path(self, model_dir: str, filename: str):

        version_dir, version = self._resolve_latest_dir(model_dir)

        self._validate_metadata(version_dir)

        path = os.path.join(version_dir, filename)

        if not os.path.exists(path):
            raise RuntimeError(f"Missing artifact: {path}")

        return path, version

    # ---------------------------------------------------
    # THREAD SAFE LOAD WRAPPER
    # ---------------------------------------------------

    def _load_once(self, attr_name, loader_func):

        if getattr(self, attr_name) is None:

            with self._lock:

                if getattr(self, attr_name) is None:
                    setattr(self, attr_name, loader_func())

        return getattr(self, attr_name)

    # ---------------------------------------------------
    # XGBOOST
    # ---------------------------------------------------

    @property
    def xgb(self):

        def load():

            path, version = self._latest_path(
                "artifacts/xgboost",
                "model.pkl"
            )

            logger.info(f"Loading XGBoost {version}")

            model = joblib.load(path)

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

            return model

        return self._load_once("_xgb", load)

    # ---------------------------------------------------
    # LSTM
    # ---------------------------------------------------

    @property
    def lstm(self):

        def load():

            path, version = self._latest_path(
                "artifacts/lstm",
                "model.keras"
            )

            logger.info(f"Loading LSTM {version}")

            model = tf.keras.models.load_model(
                path,
                compile=False
            )

            MODEL_VERSION.labels(
                model="lstm",
                version=version
            ).set(1)

            return model

        return self._load_once("_lstm", load)

    # ---------------------------------------------------

    @property
    def scaler(self):

        def load():

            path, _ = self._latest_path(
                "artifacts/lstm",
                "scaler.pkl"
            )

            logger.info("Loading LSTM scaler")

            return joblib.load(path)

        return self._load_once("_scaler", load)

    # ---------------------------------------------------
    # PROPHET
    # ---------------------------------------------------

    @property
    def prophet(self):

        def load():

            path, version = self._latest_path(
                "artifacts/prophet",
                "prophet_trend.pkl"
            )

            logger.info(f"Loading Prophet {version}")

            model = joblib.load(path)

            MODEL_VERSION.labels(
                model="prophet",
                version=version
            ).set(1)

            return model

        return self._load_once("_prophet", load)

    # ---------------------------------------------------
    # 🔥 WARMUP
    # ---------------------------------------------------

    def warmup(self):

        logger.info("🔥 Warming up models...")

        _ = self.xgb
        _ = self.lstm
        _ = self.scaler
        _ = self.prophet

        logger.info("✅ Models ready for inference.")

    # ---------------------------------------------------
    # Forecast Wrappers
    # ---------------------------------------------------

    def lstm_forecast(self, recent_prices):

        return forecast_lstm(
            self.lstm,
            self.scaler,
            recent_prices
        )

    def prophet_forecast(self):

        return forecast_prophet(self.prophet)
