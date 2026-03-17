# =========================================================
# MODEL LOADER v2.7 (STABLE + API COMPATIBLE)
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
)

from core.market.universe import MarketUniverse
from core.artifacts.metadata_manager import MetadataManager
from app.monitoring.metrics import MODEL_VERSION

logger = logging.getLogger("marketsentinel.loader")


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


class ModelLoader:

    _instance = None
    _instance_lock = threading.Lock()

    MIN_ARTIFACT_BYTES = 20_000
    MIN_METADATA_BYTES = 300

    POINTER_FILENAME = "production_pointer.json"
    BASELINE_PATH = os.path.abspath("artifacts/drift/baseline.json")

    STRICT_GOVERNANCE = os.getenv("MODEL_STRICT_GOVERNANCE", "0") == "1"
    ALLOW_POINTER_FALLBACK = os.getenv("MODEL_ALLOW_POINTER_FALLBACK", "1") == "1"

    RUNTIME_SCHEMA_SIGNATURE = get_schema_signature()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._reload_lock = threading.Lock()
        self._xgb_container: Optional[LoadedModel] = None
        self._initialized = True

    ########################################################

    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    ########################################################

    def _compute_feature_checksum(self):
        canonical = json.dumps(list(MODEL_FEATURES)).encode()
        return hashlib.sha256(canonical).hexdigest()

    ########################################################

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

            computed = hashlib.sha256(
                json.dumps(clone, sort_keys=True).encode()
            ).hexdigest()

            if integrity != computed:
                logger.warning("Baseline integrity mismatch.")
                return False, computed

            return True, computed

        except Exception:
            logger.warning("Baseline unreadable or corrupted.")
            return False, None

    ########################################################

    def _get_registry_dir(self):

        base_dir = os.getenv(
            "XGB_REGISTRY_DIR",
            os.path.abspath("artifacts/xgboost")
        )

        if not os.path.isdir(base_dir):
            raise RuntimeError("Model registry directory missing.")

        return base_dir

    ########################################################

    def _resolve_production_version(self, base_dir):

        pointer_path = os.path.join(base_dir, self.POINTER_FILENAME)

        if not os.path.exists(pointer_path):

            if not self.ALLOW_POINTER_FALLBACK:
                raise RuntimeError("Production pointer missing.")

            logger.warning("Pointer missing — using latest model.")
            return self._resolve_latest_version(base_dir)

        pointer_hash = self._sha256(pointer_path)

        with open(pointer_path, encoding="utf-8") as f:
            pointer = json.load(f)

        version = pointer.get("model_version")

        # FIX: Use pointer paths first, fall back to both naming conventions
        model_path = pointer.get("model_path")
        metadata_path = pointer.get("metadata_path")

        if not model_path or not os.path.exists(model_path):
            model_path = os.path.join(base_dir, f"{version}.joblib")

        if not metadata_path or not os.path.exists(metadata_path):
            metadata_path = os.path.join(base_dir, f"{version}_metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise RuntimeError("Pointer references missing model artifacts.")

        return model_path, metadata_path, version, pointer_hash

    ########################################################

    def _resolve_latest_version(self, base_dir):

        # FIX: Look for .joblib files matching trainer naming convention
        versions = [
            f.replace(".joblib", "")
            for f in os.listdir(base_dir)
            if f.endswith(".joblib")
        ]

        if not versions:
            raise RuntimeError("No model artifacts found.")

        version = sorted(versions)[-1]

        return (
            os.path.join(base_dir, f"{version}.joblib"),
            os.path.join(base_dir, f"{version}_metadata.json"),
            version,
            None,
        )

    ########################################################

    def _safe_load_model(self, model_path):

        if os.path.getsize(model_path) < self.MIN_ARTIFACT_BYTES:
            raise RuntimeError("Artifact too small.")

        model = joblib.load(model_path)

        if not hasattr(model, "predict"):
            raise RuntimeError("Invalid model.")

        return model

    ########################################################

    def _reload_xgb_if_needed(self):

        with self._reload_lock:

            base_dir = self._get_registry_dir()

            model_path, metadata_path, version, pointer_hash = \
                self._resolve_production_version(base_dir)

            if (
                self._xgb_container
                and self._xgb_container.version == version
                and self._xgb_container.pointer_hash == pointer_hash
            ):
                return self._xgb_container.model

            logger.info("Loading XGBoost version=%s", version)

            meta = MetadataManager.load_metadata(metadata_path)

            model = self._safe_load_model(model_path)

            baseline_available, baseline_hash = self._verify_baseline()

            self._xgb_container = LoadedModel(
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
                training_fingerprint=getattr(model, "training_fingerprint", None),
                baseline_available=baseline_available,
                baseline_hash=baseline_hash
            )

            MODEL_VERSION.labels(
                model="xgboost",
                version=version
            ).set(1)

            return model

    ########################################################
    # WARMUP FIXED
    ########################################################

    def warmup(self):

        try:
            model = self.xgb

            dummy = pd.DataFrame(
                np.zeros((1, len(MODEL_FEATURES))),
                columns=MODEL_FEATURES
            )

            model.predict(dummy)

            logger.info("Model warmup successful.")

        except Exception as exc:
            logger.warning("Model warmup skipped | %s", exc)

    ########################################################
    # PROPERTIES
    ########################################################

    @property
    def xgb(self):
        return self._reload_xgb_if_needed()

    @property
    def schema_signature(self):
        return self.RUNTIME_SCHEMA_SIGNATURE

    @property
    def xgb_version(self):
        return self._xgb_container.version if self._xgb_container else None

    @property
    def artifact_hash(self):
        return self._xgb_container.artifact_hash if self._xgb_container else None

    @property
    def dataset_hash(self):
        return self._xgb_container.dataset_hash if self._xgb_container else None

    @property
    def training_code_hash(self):
        return self._xgb_container.training_code_hash if self._xgb_container else None

    # FIX: Add missing feature_checksum property
    @property
    def feature_checksum(self):
        return self._xgb_container.feature_checksum if self._xgb_container else None

    ########################################################
    # FIX: Add missing get_feature_importance method
    ########################################################

    def get_feature_importance(self):

        model = self.xgb

        booster = getattr(model, "get_booster", None)

        if booster is not None:
            score_map = booster().get_score(importance_type="gain")
        elif hasattr(model, "feature_importances_"):
            score_map = dict(zip(MODEL_FEATURES, model.feature_importances_))
        else:
            score_map = {f: 0.0 for f in MODEL_FEATURES}

        result = []

        for feat in MODEL_FEATURES:
            result.append({
                "feature": feat,
                "importance": float(score_map.get(feat, score_map.get(f"f{list(MODEL_FEATURES).index(feat)}", 0.0))),
            })

        result.sort(key=lambda x: x["importance"], reverse=True)

        return result