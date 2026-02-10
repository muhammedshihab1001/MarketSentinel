import os
import joblib
import tensorflow as tf
import logging

from models.lstm_model import forecast_lstm
from models.prophet_model import forecast_prophet
from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Institutional Registry-Aware Model Loader.

    Guarantees:
    ✅ Loads ONLY versioned artifacts
    ✅ Exposes model version metrics
    ✅ Supports warmup loading
    ✅ Fail-fast registry validation
    ✅ Singleton per worker
    """

    _instance = None

    # ---------------------------------------------------

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    # ---------------------------------------------------

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        # TensorFlow CPU control
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        self._xgb = None
        self._lstm = None
        self._scaler = None
        self._prophet = None

        self._initialized = True

    # ---------------------------------------------------
    # INTERNAL — Resolve Latest Version
    # ---------------------------------------------------

    def _resolve_latest_dir(self, model_dir: str):

        latest_dir = os.path.join(model_dir, "latest")

        if not os.path.exists(latest_dir):
            raise RuntimeError(
                f"No 'latest' model found in {model_dir}. "
                f"Did training register the model?"
            )

        # resolve symlink → actual version
        version_dir = os.path.realpath(latest_dir)
        version = os.path.basename(version_dir)

        return version_dir, version

    # ---------------------------------------------------

    def _latest_path(self, model_dir: str, filename: str):

        version_dir, version = self._resolve_latest_dir(model_dir)

        path = os.path.join(version_dir, filename)

        if not os.path.exists(path):
            raise RuntimeError(f"Missing artifact: {path}")

        return path, version

    # ---------------------------------------------------
    # XGBOOST
    # ---------------------------------------------------

    @property
    def xgb(self):

        if self._xgb is None:

            path, version = self._latest_path(
                "artifacts/xgboost",
                "model.pkl"
            )

            logger.info(f"Loading XGBoost model from {path}")

            self._xgb = joblib.load(path)

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

        return self._xgb

    # ---------------------------------------------------
    # LSTM
    # ---------------------------------------------------

    @property
    def lstm(self):

        if self._lstm is None:

            path, version = self._latest_path(
                "artifacts/lstm",
                "model.keras"
            )

            logger.info(f"Loading LSTM model from {path}")

            self._lstm = tf.keras.models.load_model(
                path,
                compile=False
            )

            MODEL_VERSION.labels(
                model="lstm",
                version=version
            ).set(1)

        return self._lstm

    # ---------------------------------------------------

    @property
    def scaler(self):

        if self._scaler is None:

            path, _ = self._latest_path(
                "artifacts/lstm",
                "scaler.pkl"
            )

            logger.info("Loading LSTM scaler")

            self._scaler = joblib.load(path)

        return self._scaler

    # ---------------------------------------------------
    # PROPHET
    # ---------------------------------------------------

    @property
    def prophet(self):

        if self._prophet is None:

            path, version = self._latest_path(
                "artifacts/prophet",
                "prophet_trend.pkl"
            )

            logger.info(f"Loading Prophet model from {path}")

            self._prophet = joblib.load(path)

            MODEL_VERSION.labels(
                model="prophet",
                version=version
            ).set(1)

        return self._prophet

    # ---------------------------------------------------
    # 🔥 WARMUP (VERY IMPORTANT)
    # ---------------------------------------------------

    def warmup(self):
        """
        Forces model loading at startup.

        Prevents cold-start latency.
        Ensures registry validity.
        """

        logger.info("🔥 Warming up models...")

        _ = self.xgb
        _ = self.lstm
        _ = self.scaler
        _ = self.prophet

        logger.info("✅ All models loaded successfully.")

    # ---------------------------------------------------
    # Forecast Wrappers
    # ---------------------------------------------------

    def lstm_forecast(self, recent_prices):

        return forecast_lstm(
            self.lstm,
            self.scaler,
            recent_prices
        )

    # ---------------------------------------------------

    def prophet_forecast(self):

        return forecast_prophet(self.prophet)
