from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import logging
import threading

from core.schema.feature_schema import FEATURE_COUNT

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42


###################################################
# SAFE GPU DETECTION
###################################################

_GPU_AVAILABLE = None
_GPU_LOCK = threading.Lock()


def _gpu_available():
    global _GPU_AVAILABLE

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    with _GPU_LOCK:
        if _GPU_AVAILABLE is not None:
            return _GPU_AVAILABLE

        try:
            build_info = xgb.build_info()

            if not build_info.get("USE_CUDA", False):
                _GPU_AVAILABLE = False
                logger.info("XGBoost built without CUDA support.")
                return _GPU_AVAILABLE

            dtrain = xgb.DMatrix(
                np.random.rand(20, 4),
                label=np.random.randint(0, 2, 20)
            )

            xgb.train(
                {
                    "tree_method": "hist",
                    "device": "cuda",
                    "max_depth": 1,
                    "verbosity": 0
                },
                dtrain,
                num_boost_round=1
            )

            _GPU_AVAILABLE = True
            logger.info("CUDA backend verified.")

        except Exception:
            _GPU_AVAILABLE = False
            logger.info("CUDA unavailable - using CPU.")

        return _GPU_AVAILABLE


def _device():
    return "cuda" if _gpu_available() else "cpu"


###################################################
# CLASS WEIGHT
###################################################

def compute_class_weight(y):
    y = np.asarray(y)

    if len(y) == 0:
        raise RuntimeError("Empty labels passed to XGBoost.")

    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))

    if pos == 0 or neg == 0:
        raise RuntimeError("Label collapse detected.")

    weight = neg / pos
    weight = float(np.clip(weight, 0.8, 12.0))

    logger.info("Computed class weight = %.3f", weight)

    return weight


###################################################
# SAFE CLASSIFIER (FEATURE-NAME SAFE)
###################################################

class SafeXGBClassifier(XGBClassifier):

    def fit(self, X, y, **kwargs):
        """
        IMPORTANT:
        - Do NOT convert to numpy.
        - Preserve pandas column names.
        - Validate safely.
        """

        if not hasattr(X, "shape"):
            raise RuntimeError("Invalid feature matrix passed to XGBoost.")

        if X.shape[1] != FEATURE_COUNT:
            raise RuntimeError(
                f"Feature schema mismatch. Expected {FEATURE_COUNT}, got {X.shape[1]}"
            )

        # Validate finiteness safely
        if hasattr(X, "to_numpy"):
            if not np.isfinite(X.to_numpy()).all():
                raise RuntimeError("Non-finite feature values detected.")
        else:
            if not np.isfinite(X).all():
                raise RuntimeError("Non-finite feature values detected.")

        logger.info(
            "XGBoost training started | rows=%s cols=%s",
            X.shape[0],
            X.shape[1]
        )

        # Remove early stopping if accidentally passed
        kwargs.pop("early_stopping_rounds", None)

        return super().fit(X, y, **kwargs)


###################################################
# PARAM BUILDER
###################################################

def _base_params(pos_weight):

    device = _device()

    params = dict(
        n_estimators=550,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.90,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.10,
        reg_alpha=0.8,
        reg_lambda=1.5,
        tree_method="hist",
        device=device,
        max_bin=256,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=pos_weight,
        verbosity=0
    )

    logger.info(
        "XGBoost params | device=%s depth=%s trees=%s lr=%.3f",
        device.upper(),
        params["max_depth"],
        params["n_estimators"],
        params["learning_rate"]
    )

    return params


###################################################
# BUILD TRAIN MODEL
###################################################

def build_xgboost_model(y):
    pos_weight = compute_class_weight(y)
    params = _base_params(pos_weight)
    return SafeXGBClassifier(**params)


###################################################
# BUILD FINAL MODEL
###################################################

def build_final_xgboost_model(y):
    pos_weight = compute_class_weight(y)
    params = _base_params(pos_weight)

    params.update({
        "n_estimators": 700,
        "learning_rate": 0.03,
        "max_depth": 6,
        "gamma": 0.15,
        "reg_alpha": 1.0,
        "reg_lambda": 1.6,
    })

    logger.info("Building final production model.")

    return SafeXGBClassifier(**params)
