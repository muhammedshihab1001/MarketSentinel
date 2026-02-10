import os
import joblib
import tensorflow as tf
import logging

from models.lstm_model import forecast_lstm
from models.prophet_model import forecast_prophet


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Registry-aware institutional model loader.

    Guarantees:
    ✅ Loads ONLY versioned artifacts
    ✅ Prevents accidental root-model loading
    ✅ Rollback-safe
    ✅ Singleton per worker
    ✅ Lazy loading
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

        # TensorFlow thread control
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

    def _latest_path(self, model_dir: str, filename: str) -> str:
        """
        Resolves artifacts/<model>/latest/<file>

        FAILS FAST if latest is missing.
        """

        latest_dir = os.path.join(model_dir, "latest")

        if not os.path.exists(latest_dir):
            raise RuntimeError(
                f"No 'latest' model found in {model_dir}. "
                f"Did you register the model?"
            )

        path = os.path.join(latest_dir, filename)

        if not os.path.exists(path):
            raise RuntimeError(
                f"Missing artifact: {path}"
            )

        return path

    # ---------------------------------------------------
    # XGBOOST
    # ---------------------------------------------------

    @property
    def xgb(self):

        if self._xgb is None:

            path = self._latest_path(
                "artifacts/xgboost",
                "model.pkl"
            )

            logger.info(f"Loading XGBoost model from {path}")

            self._xgb = joblib.load(path)

        return self._xgb

    # ---------------------------------------------------
    # LSTM
    # ---------------------------------------------------

    @property
    def lstm(self):

        if self._lstm is None:

            path = self._latest_path(
                "artifacts/lstm",
                "model.keras"
            )

            logger.info(f"Loading LSTM model from {path}")

            self._lstm = tf.keras.models.load_model(
                path,
                compile=False
            )

        return self._lstm

    # ---------------------------------------------------

    @property
    def scaler(self):

        if self._scaler is None:

            path = self._latest_path(
                "artifacts/lstm",
                "scaler.pkl"
            )

            logger.info(f"Loading LSTM scaler from {path}")

            self._scaler = joblib.load(path)

        return self._scaler

    # ---------------------------------------------------
    # PROPHET
    # ---------------------------------------------------

    @property
    def prophet(self):

        if self._prophet is None:

            path = self._latest_path(
                "artifacts/prophet",
                "prophet_trend.pkl"
            )

            logger.info(f"Loading Prophet model from {path}")

            self._prophet = joblib.load(path)

        return self._prophet

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
