import os
import json
import joblib
import tensorflow as tf
import logging
import threading

from prophet.serialize import model_from_json

from models.lstm_model import forecast_lstm
from models.prophet_model import forecast_prophet
from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


class ModelLoader:

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

        self._configure_tensorflow()

        self._xgb = None
        self._shadow_xgb = None
        self._lstm = None
        self._scaler = None
        self._prophet = None

        self._initialized = True

    # ---------------------------------------------------
    # TENSORFLOW CONFIG
    # ---------------------------------------------------

    def _configure_tensorflow(self):

        disable_gpu = os.getenv("DISABLE_GPU", "false").lower() == "true"

        if disable_gpu:
            tf.config.set_visible_devices([], "GPU")

        intra = int(os.getenv("TF_INTRA_THREADS", "1"))
        inter = int(os.getenv("TF_INTER_THREADS", "1"))

        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------

    def _validate_artifact(self, path):

        if not os.path.exists(path):
            raise RuntimeError(f"Missing artifact: {path}")

        if os.path.getsize(path) == 0:
            raise RuntimeError(f"Empty artifact: {path}")

    # ---------------------------------------------------

    def _validate_metadata(self, version_dir):

        meta_path = os.path.join(version_dir, "metadata.json")

        if not os.path.exists(meta_path):
            raise RuntimeError("Missing metadata.json")

        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("metadata_type", "model") != "model":
            raise RuntimeError("Invalid metadata type.")

        if "schema_signature" not in meta:
            raise RuntimeError("Metadata missing schema_signature")

        if "metrics" not in meta:
            raise RuntimeError("Metadata missing metrics")

    # ---------------------------------------------------
    # POINTER RESOLUTION
    # ---------------------------------------------------

    def _resolve_production_dir(self, model_dir: str):

        pointer = os.path.join(model_dir, "latest.json")

        if os.path.exists(pointer):

            with open(pointer) as f:
                version = json.load(f)["version"]

            version_dir = os.path.join(model_dir, version)

            if not os.path.exists(version_dir):
                raise RuntimeError("Latest pointer corrupted.")

            return version_dir, version

        symlink = os.path.join(model_dir, "latest")

        if os.path.islink(symlink):

            version = os.readlink(symlink)
            version_dir = os.path.join(model_dir, version)

            return version_dir, version

        raise RuntimeError(f"No production pointer found in {model_dir}")

    # ---------------------------------------------------
    # SHADOW DISCOVERY
    # ---------------------------------------------------

    def _find_shadow_version(self, model_dir, prod_version):

        if not os.path.exists(model_dir):
            return None

        candidates = []

        for version in os.listdir(model_dir):

            if version == prod_version:
                continue

            version_dir = os.path.join(model_dir, version)

            manifest = os.path.join(version_dir, "manifest.json")

            if not os.path.exists(manifest):
                continue

            try:
                with open(manifest) as f:
                    data = json.load(f)

                if data.get("stage") == "shadow":
                    candidates.append((version_dir, version))

            except Exception:
                continue

        if not candidates:
            return None

        return sorted(candidates, key=lambda x: x[1], reverse=True)[0]

    # ---------------------------------------------------

    def _load_once(self, attr, loader):

        if getattr(self, attr) is None:

            with self._lock:

                if getattr(self, attr) is None:
                    setattr(self, attr, loader())

        return getattr(self, attr)

    # ---------------------------------------------------
    # XGBOOST
    # ---------------------------------------------------

    @property
    def xgb(self):

        def load():

            version_dir, version = self._resolve_production_dir(
                "artifacts/xgboost"
            )

            self._validate_metadata(version_dir)

            path = os.path.join(version_dir, "model.pkl")
            self._validate_artifact(path)

            logger.info(f"Loading production XGBoost {version}")

            model = joblib.load(path)

            MODEL_VERSION.labels(
                model="xgboost_prod",
                version=version
            ).set(1)

            return model

        return self._load_once("_xgb", load)

    # ---------------------------------------------------
    # SHADOW XGBOOST
    # ---------------------------------------------------

    @property
    def shadow_xgb(self):

        def load():

            prod_dir, prod_version = self._resolve_production_dir(
                "artifacts/xgboost"
            )

            result = self._find_shadow_version(
                "artifacts/xgboost",
                prod_version
            )

            if result is None:
                return None

            version_dir, version = result

            self._validate_metadata(version_dir)

            path = os.path.join(version_dir, "model.pkl")
            self._validate_artifact(path)

            logger.info(f"Loading shadow XGBoost {version}")

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
            self._validate_artifact(path)

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
            self._validate_artifact(path)

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

            path = os.path.join(version_dir, "prophet_trend.json")
            self._validate_artifact(path)

            logger.info(f"Loading Prophet {version}")

            with open(path) as f:
                model = model_from_json(f.read())

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

        logger.info("Warming production models")

        _ = self.xgb
        _ = self.lstm
        _ = self.scaler
        _ = self.prophet

        try:
            _ = self.shadow_xgb
        except Exception:
            logger.exception("Shadow warmup failed")

        logger.info("Models ready")

    # ---------------------------------------------------

    def lstm_forecast(self, recent_prices):

        return forecast_lstm(
            self.lstm,
            self.scaler,
            recent_prices
        )

    def prophet_forecast(self):

        return forecast_prophet(self.prophet)
