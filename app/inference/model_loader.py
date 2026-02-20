import os
import joblib
import logging
import threading
import json
import hashlib
from dataclasses import dataclass
from glob import glob

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES
)

from models.lstm_model import forecast_lstm
from models.sarimax_model import SarimaxModel

from app.monitoring.metrics import MODEL_VERSION


logger = logging.getLogger("marketsentinel.loader")


@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    schema_signature: str


class ModelLoader:

    _instance = None
    _lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 50_000

    ###################################################

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    ###################################################

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        self._xgb_container: LoadedModel | None = None
        self._sarimax_container: LoadedModel | None = None
        self._lstm = None
        self._scaler = None

        self._initialized = True

    ###################################################
    # HASH
    ###################################################

    def _sha256(self, path):

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

        return h.hexdigest()

    ###################################################
    # FIND LATEST TRAINING ARTIFACT
    ###################################################

    def _find_latest_artifact(self, base_dir):

        model_files = glob(os.path.join(base_dir, "model_*.pkl"))

        if not model_files:
            raise RuntimeError("No trained model artifacts found.")

        latest = max(model_files, key=os.path.getmtime)

        timestamp = os.path.basename(latest).split("_")[1].split(".")[0]

        metadata_path = os.path.join(
            base_dir,
            f"metadata_{timestamp}.json"
        )

        if not os.path.exists(metadata_path):
            raise RuntimeError("Metadata file missing for latest model.")

        return latest, metadata_path, timestamp

    ###################################################
    # VALIDATE METADATA
    ###################################################

    def _validate_metadata(self, metadata_path):

        with open(metadata_path) as f:
            meta = json.load(f)

        if meta.get("schema_signature") != get_schema_signature():
            raise RuntimeError("Schema mismatch during inference.")

        return meta

    ###################################################
    # SAFE LOAD
    ###################################################

    def _safe_load_model(self, model_path):

        if os.path.getsize(model_path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError("Artifact too small — corrupted model.")

        model = joblib.load(model_path)

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Invalid model artifact loaded.")

        return model

    ###################################################
    # XGBOOST
    ###################################################

    def _reload_xgb_if_needed(self):

        base_dir = os.getenv(
            "XGB_REGISTRY_DIR",
            os.path.abspath("artifacts/xgboost")
        )

        model_path, metadata_path, version = \
            self._find_latest_artifact(base_dir)

        if self._xgb_container and self._xgb_container.version == version:
            return self._xgb_container.model

        logger.info("Loading XGBoost version=%s", version)

        meta = self._validate_metadata(metadata_path)
        model = self._safe_load_model(model_path)

        self._xgb_container = LoadedModel(
            model=model,
            version=version,
            schema_signature=meta["schema_signature"]
        )

        MODEL_VERSION.labels(
            model="xgboost",
            version=version
        ).set(1)

        return model

    ###################################################
    # FEATURE VALIDATION AT INFERENCE
    ###################################################

    def validate_features(self, df):

        missing = set(MODEL_FEATURES) - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Inference feature mismatch: missing {missing}"
            )

        return df.loc[:, MODEL_FEATURES]

    ###################################################
    # PUBLIC ACCESSOR
    ###################################################

    @property
    def xgb(self):
        return self._reload_xgb_if_needed()

    ###################################################
    # LSTM
    ###################################################

    def _load_lstm_if_needed(self):

        if self._lstm is not None:
            return

        base_dir = os.getenv(
            "LSTM_REGISTRY_DIR",
            os.path.abspath("artifacts/lstm")
        )

        from tensorflow.keras.models import load_model

        model_path = os.path.join(base_dir, "model.keras")
        scaler_path = os.path.join(base_dir, "scalers.pkl")

        self._lstm = load_model(model_path)
        self._scaler = joblib.load(scaler_path)

        MODEL_VERSION.labels(
            model="lstm",
            version="production"
        ).set(1)

    def lstm_forecast(self, recent_prices):

        self._load_lstm_if_needed()

        return forecast_lstm(
            self._lstm,
            self._scaler,
            recent_prices
        )