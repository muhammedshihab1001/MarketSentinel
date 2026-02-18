from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import logging
import threading

from core.schema.feature_schema import FEATURE_COUNT


logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42


###################################################
# GPU DETECTION (SAFE ONCE)
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
    weight = float(np.clip(weight, 1.0, 30.0))

    logger.info("Computed class weight = %.3f", weight)

    return weight


###################################################
# FEATURE VALIDATION
###################################################

def _validate_features(X):

    if X.shape[1] != FEATURE_COUNT:
        raise RuntimeError(
            f"Feature schema mismatch. Expected {FEATURE_COUNT}, got {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite feature values detected.")


###################################################
# SAFE CLASSIFIER (EARLY STOPPING SAFE)
###################################################

class SafeXGBClassifier(XGBClassifier):

    def fit(self, X, y, **kwargs):

        X = np.asarray(X)
        _validate_features(X)

        logger.info(
            "XGBoost training started | rows=%s cols=%s",
            X.shape[0],
            X.shape[1]
        )

        return super().fit(X, y, **kwargs)


###################################################
# PARAM BUILDER
###################################################

def _base_params(pos_weight):

    device = _device()

    params = dict(

        # CORE
        n_estimators=900,
        max_depth=6,
        learning_rate=0.03,

        # STRUCTURE
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        gamma=0.20,

        # REGULARIZATION
        reg_alpha=1.0,
        reg_lambda=1.5,

        # SAFETY
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        max_bin=256,

        # TREE
        tree_method="hist",
        device=device,

        # IMBALANCE
        scale_pos_weight=pos_weight,

        verbosity=0
    )

    logger.info(
        "XGBoost params | %s hist | lr=%.3f depth=%s",
        device.upper(),
        params["learning_rate"],
        params["max_depth"]
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
# BUILD FINAL PRODUCTION MODEL
###################################################

def build_final_xgboost_model(y):

    pos_weight = compute_class_weight(y)
    params = _base_params(pos_weight)

    params.update({
        "n_estimators": 1100,
        "learning_rate": 0.02,
        "gamma": 0.25,
        "reg_alpha": 1.2,
        "reg_lambda": 1.8,
        "max_depth": 7,
    })

    logger.info("Building final production model.")

    return SafeXGBClassifier(**params)
