import os
import json
import joblib
import tensorflow as tf
import logging
import threading
import hashlib

from datetime import datetime

from prophet.serialize import model_from_json

from models.lstm_model import forecast_lstm
from models.prophet_model import forecast_prophet

from core.schema.feature_schema import get_schema_signature
from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


class ModelLoader:
    """
    Institutional Registry-Aware Model Loader.

    Guarantees:
    - manifest hash verification
    - schema lock enforcement
    - lineage validation
    - pointer crash safety
    - atomic hot reload
    - inference boundary protection
    """

    _instance = None
    _instance_lock = threading.Lock()
    _tf_lock = threading.Lock()

    # ---------------------------------------------------

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    # ---------------------------------------------------

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        self._configure_tensorflow()

        self._xgb = None
        self._xgb_version = None

        self._shadow_xgb = None

        self._lstm = None
        self._scaler = None
        self._prophet = None

        self._load_lock = threading.Lock()

        self._initialized = True

    # ---------------------------------------------------
    # SAFE TF INIT
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
    # HASH
    # ---------------------------------------------------

    @staticmethod
    def _sha256(path: str):

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    # ---------------------------------------------------
    # ARTIFACT VALIDATION
    # ---------------------------------------------------

    def _validate_artifact(self, path):

        if not os.path.exists(path):
            raise RuntimeError(f"Missing artifact: {path}")

        if os.path.getsize(path) == 0:
            raise RuntimeError(f"Empty artifact: {path}")

    # ---------------------------------------------------
    # MANIFEST + HASH VERIFICATION
    # ---------------------------------------------------

    def _validate_manifest_and_hash(self, version_dir):

        manifest_path = os.path.join(version_dir, "manifest.json")

        if not os.path.exists(manifest_path):
            raise RuntimeError("Manifest missing.")

        with open(manifest_path) as f:
            manifest = json.load(f)

        if manifest.get("stage") != "production":
            raise RuntimeError(
                "Attempted to load non-production model."
            )

        artifacts = manifest.get("artifacts", {})

        for artifact, expected_hash in artifacts.items():

            path = os.path.join(version_dir, artifact)

            self._validate_artifact(path)

            actual = self._sha256(path)

            if actual != expected_hash:
                raise RuntimeError(
                    f"Artifact integrity failure: {artifact}"
                )

        return manifest

    # ---------------------------------------------------
    # METADATA VALIDATION
    # ---------------------------------------------------

    def _validate_metadata(self, version_dir):

        meta_path = os.path.join(version_dir, "metadata.json")

        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("schema_signature") != get_schema_signature():
            raise RuntimeError("Schema mismatch detected.")

        if "training_code_hash" not in meta:
            raise RuntimeError("Training lineage missing.")

        return meta

    # ---------------------------------------------------
    # POINTER RESOLUTION (CRASH SAFE)
    # ---------------------------------------------------

    def _resolve_production_dir(self, model_dir):

        pointer = os.path.join(model_dir, "latest.json")

        if not os.path.exists(pointer):
            raise RuntimeError("Latest pointer missing.")

        with open(pointer) as f:
            version = json.load(f)["version"]

        version_dir = os.path.join(model_dir, version)

        if not os.path.exists(version_dir):
            raise RuntimeError(
                "Registry pointer corrupted. Version directory missing."
            )

        self._validate_manifest_and_hash(version_dir)

        return version_dir, version

    # ---------------------------------------------------
    # HOT RELOAD
    # ---------------------------------------------------

    def _reload_if_needed(
        self,
        attr,
        version_attr,
        loader,
        current_version
    ):

        cached_version = getattr(self, version_attr)

        if cached_version == current_version:
            return getattr(self, attr)

        with self._load_lock:

            cached_version = getattr(self, version_attr)

            if cached_version != current_version:

                logger.warning(
                    f"Reloading model due to version change → {current_version}"
                )

                model = loader()

                setattr(self, attr, model)
                setattr(self, version_attr, current_version)

        return getattr(self, attr)

    # ---------------------------------------------------
    # XGBOOST
    # ---------------------------------------------------

    @property
    def xgb(self):

        version_dir, version = self._resolve_production_dir(
            "artifacts/xgboost"
        )

        def load():

            self._validate_metadata(version_dir)

            path = os.path.join(version_dir, "model.pkl")

            model = joblib.load(path)

            if not hasattr(model, "predict_proba"):
                raise RuntimeError("Invalid XGBoost artifact.")

            MODEL_VERSION.labels(
                model="xgboost_prod",
                version=version
            ).set(1)

            return model

        return self._reload_if_needed(
            "_xgb",
            "_xgb_version",
            load,
            version
        )

    # ---------------------------------------------------
    # SHADOW MODEL
    # ---------------------------------------------------

    @property
    def shadow_xgb(self):

        prod_dir, prod_version = self._resolve_production_dir(
            "artifacts/xgboost"
        )

        model_dir = "artifacts/xgboost"

        shadow = None
        newest_ts = None

        for v in os.listdir(model_dir):

            if v == prod_version:
                continue

            manifest = os.path.join(
                model_dir,
                v,
                "manifest.json"
            )

            if not os.path.exists(manifest):
                continue

            with open(manifest) as f:
                m = json.load(f)

            if m.get("stage") != "shadow":
                continue

            ts = datetime.strptime(v[1:16], "%Y_%m_%d_%H%M")

            if newest_ts is None or ts > newest_ts:
                newest_ts = ts
                shadow = v

        if shadow is None:
            raise RuntimeError("No shadow model available.")

        shadow_dir = os.path.join(model_dir, shadow)

        self._validate_manifest_and_hash(shadow_dir)
        self._validate_metadata(shadow_dir)

        path = os.path.join(shadow_dir, "model.pkl")

        return joblib.load(path)

    # ---------------------------------------------------
    # OPTIONAL WARMUP
    # ---------------------------------------------------

    def warmup(self):

        if os.getenv("MODEL_WARMUP", "true") != "true":
            return

        logger.info("Warming production models")

        _ = self.xgb

        try:
            _ = self.shadow_xgb
        except Exception:
            pass

        logger.info("Models ready")

    # ---------------------------------------------------

    def lstm_forecast(self, recent_prices):

        return forecast_lstm(
            self._lstm,
            self._scaler,
            recent_prices
        )

    def prophet_forecast(self):

        return forecast_prophet(self._prophet)
