import os
import joblib
import tensorflow as tf
import logging

from models.lstm_model import forecast_lstm
from models.prophet_model import forecast_prophet


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Institutional-grade process-safe lazy model loader.

    Guarantees:
    ✅ Single instance per worker
    ✅ Lazy artifact loading
    ✅ Prevents accidental memory duplication
    ✅ TensorFlow thread control
    ✅ Future pre-fork compatible
    """

    _instance = None  # process-level singleton

    # ---------------------------------------------------

    def __new__(cls, *args, **kwargs):
        """
        Ensures only ONE loader exists per process.
        Critical for memory stability.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    # ---------------------------------------------------

    def __init__(self):

        # Prevent reinitialization
        if hasattr(self, "_initialized"):
            return

        # -------- Thread Safety (VERY IMPORTANT) --------
        # Prevent TF from consuming all CPU cores
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        # Optional but recommended for many containers
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # ------------------------------------------------

        self._xgb = None
        self._lstm = None
        self._scaler = None
        self._prophet = None

        self._initialized = True

    # ---------------------------------------------------
    # XGBoost
    # ---------------------------------------------------

    @property
    def xgb(self):

        if self._xgb is None:
            logger.info("Loading XGBoost model...")
            self._xgb = joblib.load("artifacts/xgboost/model.pkl")

        return self._xgb

    # ---------------------------------------------------
    # LSTM
    # ---------------------------------------------------

    @property
    def lstm(self):

        if self._lstm is None:
            logger.info("Loading LSTM model...")

            self._lstm = tf.keras.models.load_model(
                "artifacts/lstm/model.keras",
                compile=False  # 🔥 avoids unnecessary graph compile
            )

        return self._lstm

    # ---------------------------------------------------

    @property
    def scaler(self):

        if self._scaler is None:
            logger.info("Loading LSTM scaler...")
            self._scaler = joblib.load(
                "artifacts/lstm/scaler.pkl"
            )

        return self._scaler

    # ---------------------------------------------------
    # Prophet
    # ---------------------------------------------------

    @property
    def prophet(self):

        if self._prophet is None:
            logger.info("Loading Prophet model...")
            self._prophet = joblib.load(
                "artifacts/prophet/model.pkl"
            )

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
