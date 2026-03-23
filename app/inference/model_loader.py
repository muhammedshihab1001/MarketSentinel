# =========================================================
# MODEL LOADER v2.8
# FIX: Scan for .pkl artifacts (not .joblib — models are
#      saved as .pkl by train_xgboost.py export_artifacts).
# FIX: Fallback pointer file scan also looks for .pkl.
# FIX: Added safe None-check before loading artifact.
# =========================================================

import os
import joblib
import logging
import threading
import hashlib
import json
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger("marketsentinel.model_loader")

_MODEL_LOCK = threading.Lock()
_LOADED_MODEL = None
_LOADED_VERSION = None
_LOADED_HASH = None


class ModelLoader:
    """
    Thread-safe model loader for the inference API.

    Scans artifacts/xgboost/ for the latest .pkl model file.
    Supports pointer-file fallback for promoted model tracking.
    """

    # FIX: Extension is .pkl — train_xgboost saves with joblib but
    # names the file model_xgb_<version>.pkl
    MODEL_EXTENSION = ".pkl"
    POINTER_FILENAME = "latest.json"

    def __init__(self):
        self.registry_dir = os.getenv("XGB_REGISTRY_DIR", "artifacts/xgboost")
        self._model = None
        self._version = None
        self._artifact_hash = None
        self._schema_signature = None
        self._feature_names = None
        self._metadata = None

    # =====================================================
    # FIND LATEST MODEL
    # =====================================================

    def _find_via_pointer(self) -> Optional[str]:
        """
        Try to find model path via latest.json pointer file.
        Returns absolute path or None.
        """
        pointer_path = os.path.join(self.registry_dir, self.POINTER_FILENAME)

        if not os.path.exists(pointer_path):
            return None

        try:
            with open(pointer_path, encoding="utf-8") as f:
                data = json.load(f)

            path = data.get("path") or data.get("model_path")

            if not path:
                return None

            # Resolve relative paths
            if not os.path.isabs(path):
                path = os.path.join(self.registry_dir, path)

            if os.path.exists(path):
                logger.info("Model pointer found: %s", path)
                return path

        except Exception as e:
            logger.warning("Pointer file read failed: %s", e)

        return None

    def _find_via_scan(self) -> Optional[str]:
        """
        Scan the registry dir for the latest .pkl model file.
        Sorts by filename (which includes timestamp) descending.
        Returns absolute path or None.
        """
        if not os.path.isdir(self.registry_dir):
            logger.warning("Registry dir missing: %s", self.registry_dir)
            return None

        # FIX: Look for .pkl files, not .joblib
        candidates = [
            f for f in os.listdir(self.registry_dir)
            if f.endswith(self.MODEL_EXTENSION)
            and f.startswith("model_")
        ]

        if not candidates:
            logger.warning(
                "No %s model files found in %s",
                self.MODEL_EXTENSION,
                self.registry_dir,
            )
            return None

        # Sort descending — latest timestamp is last alphabetically
        candidates.sort(reverse=True)
        latest = candidates[0]
        path = os.path.join(self.registry_dir, latest)

        logger.info("Model found via scan: %s", path)
        return path

    def _find_model_path(self) -> Optional[str]:
        """
        Find model path — pointer file first, then directory scan.
        """
        allow_fallback = os.getenv("MODEL_ALLOW_POINTER_FALLBACK", "1") == "1"

        # Try pointer file first
        path = self._find_via_pointer()
        if path:
            return path

        if not allow_fallback:
            logger.error("Pointer file missing and fallback disabled.")
            return None

        # Fallback to directory scan
        return self._find_via_scan()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def _compute_artifact_hash(self, path: str) -> str:
        """Compute SHA256 hash of the artifact file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_metadata(self, model_path: str) -> Optional[dict]:
        """
        Try to load companion metadata.json for the model artifact.
        Returns None if not found — metadata is optional.
        """
        # Try same dir, same stem with .json extension
        stem = os.path.splitext(model_path)[0]
        candidates = [
            stem + ".json",
            stem.replace("model_xgb_", "metadata_") + ".json",
            os.path.join(os.path.dirname(model_path), "metadata.json"),
        ]

        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning("Metadata load failed: %s", e)

        return None

    def load(self) -> bool:
        """
        Load the latest model from the registry.
        Thread-safe. Returns True on success.
        """
        global _LOADED_MODEL, _LOADED_VERSION, _LOADED_HASH

        with _MODEL_LOCK:
            model_path = self._find_model_path()

            if model_path is None:
                logger.error("No model artifact found — cannot load.")
                return False

            try:
                logger.info("Loading model from: %s", model_path)

                artifact = joblib.load(model_path)

                # FIX: Safe None check before attribute access
                if artifact is None:
                    logger.error("Loaded artifact is None: %s", model_path)
                    return False

                self._model = artifact
                self._artifact_hash = self._compute_artifact_hash(model_path)

                # Extract version from filename: model_xgb_20260319_014404.pkl
                filename = os.path.basename(model_path)
                stem = os.path.splitext(filename)[0]  # model_xgb_20260319_014404
                parts = stem.split("_", 2)             # ['model', 'xgb', '20260319_014404']
                self._version = "_".join(parts[1:]) if len(parts) >= 3 else stem

                # Load feature names from model if available
                if hasattr(artifact, "feature_names"):
                    self._feature_names = list(artifact.feature_names)
                elif hasattr(artifact, "model") and hasattr(artifact.model, "feature_names"):
                    self._feature_names = list(artifact.model.feature_names)

                # Load schema signature
                if hasattr(artifact, "schema_signature"):
                    self._schema_signature = artifact.schema_signature
                else:
                    try:
                        from core.schema.feature_schema import get_schema_signature
                        self._schema_signature = get_schema_signature()
                    except Exception:
                        self._schema_signature = "unknown"

                # Load companion metadata
                self._metadata = self._load_metadata(model_path)

                # Update global cache
                _LOADED_MODEL = self._model
                _LOADED_VERSION = self._version
                _LOADED_HASH = self._artifact_hash

                logger.info(
                    "Model loaded | version=%s | hash=%s...",
                    self._version,
                    self._artifact_hash[:12],
                )
                return True

            except Exception as e:
                logger.exception("Model load failed: %s", e)
                return False

    # =====================================================
    # ACCESSORS
    # =====================================================

    @property
    def model(self):
        return self._model

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def artifact_hash(self) -> Optional[str]:
        return self._artifact_hash

    @property
    def schema_signature(self) -> Optional[str]:
        return self._schema_signature

    @property
    def feature_names(self) -> Optional[list]:
        return self._feature_names

    @property
    def metadata(self) -> Optional[dict]:
        return self._metadata

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Run inference. Delegates to the loaded model's predict method.
        Raises RuntimeError if model not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self._model.predict(X)

    def get_info(self) -> dict:
        """Return model metadata dict for /model/info endpoint."""
        meta = self._metadata or {}

        return {
            "model_version": self._version or "unknown",
            "schema_signature": self._schema_signature or meta.get("schema_signature", "unknown"),
            "artifact_hash": self._artifact_hash or "unknown",
            "dataset_hash": meta.get("dataset_hash", "unknown"),
            "training_code_hash": meta.get("training_code_hash", "unknown"),
            "feature_checksum": meta.get("feature_checksum", "unknown"),
            "feature_count": len(self._feature_names) if self._feature_names else 0,
        }


# =========================================================
# MODULE-LEVEL SINGLETON
# Used by app/main.py and inference pipeline
# =========================================================

_loader_instance: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Return the singleton ModelLoader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ModelLoader()
    return _loader_instance


def load_model() -> bool:
    """Load (or reload) the model. Returns True on success."""
    return get_model_loader().load()


def get_model():
    """Return the loaded model object or None."""
    loader = get_model_loader()
    return loader.model if loader.is_loaded() else None