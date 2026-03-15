# =========================================================
# MODEL LOADER v2.4
# Hybrid Multi-Agent Compatible | CV-Optimized Governance
# Pointer-Fallback Enabled | Baseline Verified
# =========================================================

import os
import joblib
import logging
import threading
import hashlib
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from core.schema.feature_schema import (
    get_schema_signature,
    MODEL_FEATURES,
    SCHEMA_VERSION,
)

from core.market.universe import MarketUniverse
from core.artifacts.metadata_manager import MetadataManager
from app.monitoring.metrics import MODEL_VERSION

logger = logging.getLogger("marketsentinel.loader")


# =========================================================
# LOADED MODEL CONTAINER
# =========================================================

@dataclass(frozen=True)
class LoadedModel:

    model: object

    version: str

    schema_signature: str

    schema_version: str

    dataset_hash: str

    artifact_hash: str

    feature_checksum: str

    universe_hash: str

    training_code_hash: Optional[str]

    reproducibility_hash: Optional[str]

    pointer_hash: Optional[str]

    training_fingerprint: Optional[str]

    baseline_available: bool

    baseline_hash: Optional[str]


# =========================================================
# MODEL LOADER (Singleton)
# =========================================================

class ModelLoader:

    _instance = None
    _instance_lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 20_000
    MIN_METADATA_BYTES = 300

    POINTER_FILENAME = "production_pointer.json"

    BASELINE_PATH = os.path.abspath("artifacts/drift/baseline.json")

    STRICT_GOVERNANCE = os.getenv("MODEL_STRICT_GOVERNANCE", "0") == "1"

    ALLOW_POINTER_FALLBACK = os.getenv("MODEL_ALLOW_POINTER_FALLBACK", "1") == "1"

    # -----------------------------------------------------

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:

            with cls._instance_lock:

                if cls._instance is None:

                    cls._instance = super().__new__(cls)

        return cls._instance

    # -----------------------------------------------------

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        self._reload_lock = threading.Lock()

        self._xgb_container: Optional[LoadedModel] = None

        self._initialized = True

    # =====================================================
    # HASH UTIL
    # =====================================================

    def _sha256(self, path: str) -> str:

        h = hashlib.sha256()

        with open(path, "rb") as f:

            for chunk in iter(lambda: f.read(1 << 20), b""):

                h.update(chunk)

        return h.hexdigest()

    # =====================================================
    # FEATURE CHECKSUM
    # =====================================================

    def _compute_feature_checksum(self):

        canonical = json.dumps(
            list(MODEL_FEATURES),
            separators=(",", ":")
        ).encode()

        return hashlib.sha256(canonical).hexdigest()

    # =====================================================
    # BASELINE VERIFICATION
    # =====================================================

    def _verify_baseline(self):

        if not os.path.exists(self.BASELINE_PATH):

            logger.warning("Baseline file missing.")

            return False, None

        try:

            with open(self.BASELINE_PATH, encoding="utf-8") as f:

                baseline = json.load(f)

            integrity = baseline.get("integrity_hash")

            clone = dict(baseline)

            clone.pop("integrity_hash", None)

            canonical = json.dumps(clone, sort_keys=True).encode()

            computed = hashlib.sha256(canonical).hexdigest()

            if integrity != computed:

                logger.warning("Baseline integrity mismatch.")

                return False, computed

            return True, computed

        except Exception:

            logger.warning("Baseline unreadable or corrupted.")

            return False, None

    # =====================================================
    # REGISTRY
    # =====================================================

    def _get_registry_dir(self):

        base_dir = os.getenv(
            "XGB_REGISTRY_DIR",
            os.path.abspath("artifacts/xgboost")
        )

        if not os.path.isdir(base_dir):
            raise RuntimeError("Model registry directory missing.")

        return base_dir

    # =====================================================
    # POINTER RESOLUTION
    # =====================================================

    def _resolve_production_version(self, base_dir) -> Tuple[str, str, str, Optional[str]]:

        pointer_path = os.path.join(base_dir, self.POINTER_FILENAME)

        if not os.path.exists(pointer_path):

            if not self.ALLOW_POINTER_FALLBACK:
                raise RuntimeError("Production pointer missing.")

            logger.warning(
                "Production pointer missing — falling back to latest model."
            )

            return self._resolve_latest_version(base_dir)

        pointer_hash = self._sha256(pointer_path)

        with open(pointer_path, encoding="utf-8") as f:

            pointer = json.load(f)

        version = pointer.get("model_version")

        if not version:
            raise RuntimeError("Invalid production pointer format.")

        model_path = os.path.join(base_dir, f"model_{version}.pkl")

        metadata_path = os.path.join(base_dir, f"metadata_{version}.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):

            raise RuntimeError("Pointer references missing model artifacts.")

        return model_path, metadata_path, version, pointer_hash

    # -----------------------------------------------------

    def _resolve_latest_version(self, base_dir):

        files = os.listdir(base_dir)

        versions = []

        for f in files:

            if f.startswith("model_") and f.endswith(".pkl"):

                version = f.replace("model_", "").replace(".pkl", "")

                versions.append(version)

        if not versions:
            raise RuntimeError("No model artifacts found.")

        # Versions are timestamp-based (e.g. "20240115_143022")
        # which sort correctly as plain strings — newest is last
        latest_version = sorted(versions)[-1]

        model_path = os.path.join(base_dir, f"model_{latest_version}.pkl")

        metadata_path = os.path.join(base_dir, f"metadata_{latest_version}.json")

        if not os.path.exists(metadata_path):

            raise RuntimeError("Metadata missing for latest model.")

        return model_path, metadata_path, latest_version, None

    # =====================================================
    # SAFE MODEL LOAD
    # =====================================================

    def _safe_load_model(self, model_path):

        if os.path.getsize(model_path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError("Artifact too small — corrupted.")

        model = joblib.load(model_path)

        if not hasattr(model, "predict"):
            raise RuntimeError("Invalid model artifact.")

        if hasattr(model, "feature_names"):

            if list(model.feature_names) != list(MODEL_FEATURES):

                msg = "Model feature order mismatch."

                if self.STRICT_GOVERNANCE:
                    raise RuntimeError(msg)

                logger.warning(msg)

        return model

    # =====================================================
    # RELOAD LOGIC
    # =====================================================

    def _reload_xgb_if_needed(self):

        with self._reload_lock:

            base_dir = self._get_registry_dir()

            model_path, metadata_path, version, pointer_hash = \
                self._resolve_production_version(base_dir)

            if (
                self._xgb_container and
                self._xgb_container.version == version and
                self._xgb_container.pointer_hash == pointer_hash
            ):
                return self._xgb_container.model

            logger.info("Loading XGBoost version=%s", version)

            if os.path.getsize(metadata_path) < self.MIN_METADATA_BYTES:
                raise RuntimeError("Metadata file corrupted.")

            meta = MetadataManager.load_metadata(metadata_path)

            actual_hash = self._sha256(model_path)

            if meta.get("artifact_hash") != actual_hash:
                raise RuntimeError("Artifact tampering detected.")

            model = self._safe_load_model(model_path)

            baseline_available, baseline_hash = self._verify_baseline()

            training_fingerprint = getattr(
                model,
                "training_fingerprint",
                None
            )

            new_container = LoadedModel(
                model=model,
                version=version,
                schema_signature=meta.get("schema_signature"),
                schema_version=meta.get("schema_version"),
                dataset_hash=meta.get("dataset_hash"),
                artifact_hash=meta.get("artifact_hash"),
                feature_checksum=meta.get("feature_checksum"),
                universe_hash=meta.get("universe_hash"),
                training_code_hash=meta.get("training_code_hash"),
                reproducibility_hash=meta.get("reproducibility_hash"),
                pointer_hash=pointer_hash,
                training_fingerprint=training_fingerprint,
                baseline_available=baseline_available,
                baseline_hash=baseline_hash
            )

            self._xgb_container = new_container

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

            logger.info(
                "Model loaded | version=%s | baseline=%s",
                version,
                "OK" if baseline_available else "MISSING"
            )

            return new_container.model

    # =====================================================
    # PUBLIC ACCESSORS
    # =====================================================

    @property
    def xgb(self):
        return self._reload_xgb_if_needed()

    @property
    def xgb_version(self):
        self._reload_xgb_if_needed()
        return getattr(self._xgb_container, "version", None)

    @property
    def schema_signature(self):
        self._reload_xgb_if_needed()
        return getattr(self._xgb_container, "schema_signature", None)

    @property
    def dataset_hash(self):
        self._reload_xgb_if_needed()
        return getattr(self._xgb_container, "dataset_hash", None)

    @property
    def artifact_hash(self):
        self._reload_xgb_if_needed()
        return getattr(self._xgb_container, "artifact_hash", None)

    @property
    def training_code_hash(self):
        self._reload_xgb_if_needed()
        return getattr(self._xgb_container, "training_code_hash", None)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    def get_feature_importance(self):

        model = self.xgb

        if not hasattr(model, "export_feature_importance"):
            raise RuntimeError(
                "Model does not support feature importance export."
            )

        return model.export_feature_importance()

    # =====================================================
    # WARMUP
    # =====================================================

    def warmup(self):

        logger.info("Model warmup triggered.")

        model = self.xgb

        try:
            # SafeXGBRegressor.predict() expects a DataFrame with
            # named columns matching MODEL_FEATURES, not a numpy array.
            dummy = pd.DataFrame(
                np.zeros((1, len(MODEL_FEATURES)), dtype=np.float32),
                columns=MODEL_FEATURES,
            )

            model.predict(dummy)

            logger.info("Model warmup inference OK.")

        except Exception as exc:

            logger.warning("Model warmup inference failed (non-critical): %s", exc)