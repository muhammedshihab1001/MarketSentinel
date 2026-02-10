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
    Stage-Aware Institutional Model Loader.

    Guarantees:
    - production pointer authority
    - shadow model capability
    - thread-safe lazy loading
    - metadata validation
    - cold-start protection
    """

    _instance = None
    _lock = threading.Lock()

    # ---------------------------------------------------

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ---------------------------------------------------

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        self._xgb = None
        self._shadow_xgb = None

        self._lstm = None
        self._scaler = None
        self._prophet = None

        self._initialized = True

    # ---------------------------------------------------
    # REGISTRY RESOLUTION
    # ---------------------------------------------------

    def _resolve_production_dir(self, model_dir: str):

        latest_link = os.path.join(model_dir, "latest")

        if not os.path.islink(latest_link):
            raise RuntimeError(
                f"No production pointer in {model_dir}"
            )

        version = os.readlink(latest_link)

        version_dir = os.path.join(model_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError("Production pointer corrupted.")

        return version_dir, version

    # ---------------------------------------------------
    # SHADOW DISCOVERY
    # ---------------------------------------------------

    def _find_shadow_version(self, model_dir: str):

        if not os.path.exists(model_dir):
            return None

        for version in sorted(os.listdir(model_dir), reverse=True):

            version_dir = os.path.join(model_dir, version)

            manifest = os.path.join(
                version_dir,
                "manifest.json"
            )

            if not os.path.exists(manifest):
                continue

            try:
                with open(manifest) as f:
                    data = json.load(f)

                if data.get("stage") in ("shadow", "approved"):
                    return version_dir, version

            except Exception:
                continue

        return None

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

        if float(schema.split(".")[0]) < 2:
            raise RuntimeError(
                "Outdated model schema. Retrain required."
            )

    # ---------------------------------------------------

    def _load_once(self, attr_name, loader_func):

        if getattr(self, attr_name) is None:

            with self._lock:

                if getattr(self, attr_name) is None:
                    setattr(self, attr_name, loader_func())

        return getattr(self, attr_name)

    # ---------------------------------------------------
    # XGBOOST — PRODUCTION
    # ---------------------------------------------------

    @property
    def xgb(self):

        def load():

            version_dir, version = self._resolve_production_dir(
                "artifacts/xgboost"
            )

            self._validate_metadata(version_dir)

            path = os.path.join(version_dir, "model.pkl")

            logger.info(f"Loading PRODUCTION XGBoost {version}")

            model = joblib.load(path)

            MODEL_VERSION.labels(
                model="xgboost_prod",
                version=version
            ).set(1)

            return model

        return self._load_once("_xgb", load)

    # ---------------------------------------------------
    # XGBOOST — SHADOW
    # ---------------------------------------------------

    @property
    def shadow_xgb(self):

        def load():

            result = self._find_shadow_version(
                "artifacts/xgboost"
            )

            if result is None:
                return None

            version_dir, version = result

            self._validate_metadata(version_dir)

            path = os.path.join(version_dir, "model.pkl")

            logger.info(f"Loading SHADOW XGBoost {version}")

            model = joblib.load(path)

            MODEL_VERSION.labels(
                model="xgboost_shadow",
                version=version
            ).set(1)

            return model

        return self._load_once("_shadow_xgb", load)

    # ---------------------------------------------------
    # LSTM
    # ---------------------------------------------------

    @property
    def lstm(self):

        def load():

            version_dir, version = self._resolve_production_dir(
                "artifacts/lstm"
            )

            self._validate_metadata(version_dir)

            path = os.path.join(version_dir, "model.keras")

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

            version_dir, _ = self._resolve_production_dir(
                "artifacts/lstm"
            )

            path = os.path.join(version_dir, "scaler.pkl")

            return joblib.load(path)

        return self._load_once("_scaler", load)

    # ---------------------------------------------------
    # PROPHET
    # ---------------------------------------------------

    @property
    def prophet(self):

        def load():

            version_dir, version = self._resolve_production_dir(
                "artifacts/prophet"
            )

            self._validate_metadata(version_dir)

            path = os.path.join(version_dir, "prophet_trend.pkl")

            logger.info(f"Loading Prophet {version}")

            model = joblib.load(path)

            MODEL_VERSION.labels(
                model="prophet",
                version=version
            ).set(1)

            return model

        return self._load_once("_prophet", load)

    # ---------------------------------------------------
    # WARMUP
    # ---------------------------------------------------

    def warmup(self):

        logger.info("Warming production models...")

        _ = self.xgb
        _ = self.lstm
        _ = self.scaler
        _ = self.prophet

        # shadow is optional
        try:
            _ = self.shadow_xgb
        except Exception:
            pass

        logger.info("Models ready.")

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
