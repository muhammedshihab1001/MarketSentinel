from xgboost import XGBClassifier
import numpy as np
import xgboost as xgb
import threading
import logging

from core.schema.feature_schema import FEATURE_COUNT


logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42


###################################################
# GPU DETECTION
###################################################

_GPU_AVAILABLE = None
_GPU_LOCK = threading.Lock()


def _gpu_verified():

    global _GPU_AVAILABLE

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    with _GPU_LOCK:

        if _GPU_AVAILABLE is not None:
            return _GPU_AVAILABLE

        try:

            dtrain = xgb.DMatrix(
                np.random.rand(50, 4),
                label=np.random.randint(0, 2, 50)
            )

            params = {
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 1,
                "verbosity": 0
            }

            xgb.train(params, dtrain, num_boost_round=1)

            _GPU_AVAILABLE = True
            logger.info("GPU detected.")

        except Exception:
            _GPU_AVAILABLE = False
            logger.info("GPU not available - using CPU.")

        return _GPU_AVAILABLE


def _device():
    return "cuda" if _gpu_verified() else "cpu"


###################################################
# CLASS WEIGHT
###################################################

def compute_class_weight(y):

    y = np.asarray(y)

    pos = float(np.sum(y))
    neg = float(len(y) - pos)

    if pos == 0 or neg == 0:
        raise RuntimeError("Label collapse detected.")

    weight = neg / pos

    logger.info("Computed class weight - %.3f", weight)

    return float(np.clip(weight, 1.0, 30.0))


###################################################
# FEATURE SAFETY
###################################################

def _validate_features(X):

    if X.shape[1] != FEATURE_COUNT:
        raise RuntimeError(
            f"Feature schema mismatch. Expected {FEATURE_COUNT}, got {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite feature values.")


###################################################
# SAFE CLASSIFIER (XGBoost 2.x Compatible)
###################################################

class SafeXGBClassifier(XGBClassifier):

    def fit(self, X, y, **kwargs):

        X = np.asarray(X)
        _validate_features(X)

        logger.info(
            "XGBoost training started | rows=%s",
            len(X)
        )

        # 🚨 DO NOT PASS unsupported kwargs
        kwargs.pop("early_stopping_rounds", None)

        return super().fit(X, y, **kwargs)


###################################################
# PARAM BUILDER
###################################################

def _base_params(pos_weight):

    device = _device()

    params = dict(

        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,

        subsample=0.8,
        colsample_bytree=0.8,

        min_child_weight=3,
        gamma=0.15,

        reg_alpha=0.7,
        reg_lambda=1.2,

        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,

        tree_method="hist",
        device=device,

        scale_pos_weight=pos_weight,

        verbosity=0
    )

    logger.info(
        "XGBoost params | device=%s lr=%.3f",
        device.upper(),
        params["learning_rate"]
    )

    return params


###################################################
# BUILD MODELS
###################################################

def build_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight)

    return SafeXGBClassifier(**params)


def build_final_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight)

    params.update({
        "n_estimators": 800,
        "learning_rate": 0.025,
        "gamma": 0.20,
        "reg_alpha": 0.9,
        "reg_lambda": 1.4,
    })

    return SafeXGBClassifier(**params)
