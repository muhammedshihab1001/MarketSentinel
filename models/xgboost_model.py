from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import logging
import threading

from core.schema.feature_schema import FEATURE_COUNT

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42


###################################################
# ROBUST GPU DETECTION
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
                np.random.rand(16, 4),
                label=np.random.randint(0, 2, 16)
            )

            xgb.train(
                {
                    "tree_method": "gpu_hist",
                    "max_depth": 1,
                    "verbosity": 0
                },
                dtrain,
                num_boost_round=1
            )

            _GPU_AVAILABLE = True
            logger.info("CUDA backend verified successfully.")

        except Exception:
            _GPU_AVAILABLE = False
            logger.info("CUDA unavailable — using CPU.")

        return _GPU_AVAILABLE


def _tree_method():
    return "gpu_hist" if _gpu_available() else "hist"


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
    weight = float(np.clip(weight, 0.5, 15.0))

    logger.info("Computed class weight = %.3f", weight)

    return weight


###################################################
# SAFE CLASSIFIER WITH EARLY STOPPING
###################################################

class SafeXGBClassifier(XGBClassifier):

    def fit(self, X, y, **kwargs):

        if not hasattr(X, "shape"):
            raise RuntimeError("Invalid feature matrix passed to XGBoost.")

        if X.shape[1] != FEATURE_COUNT:
            raise RuntimeError(
                f"Feature schema mismatch. Expected {FEATURE_COUNT}, got {X.shape[1]}"
            )

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

        return super().fit(X, y, **kwargs)


###################################################
# PARAM BUILDER (ALPHA DISCOVERY MODE)
###################################################

def _base_params(pos_weight):

    tree_method = _tree_method()

    params = dict(
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.05,

        subsample=0.80,
        colsample_bytree=0.70,
        colsample_bylevel=0.70,

        min_child_weight=2,
        gamma=0.0,

        reg_alpha=0.5,
        reg_lambda=1.0,

        max_bin=384,

        tree_method=tree_method,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=pos_weight,
        verbosity=0
    )

    logger.info(
        "XGBoost params | tree_method=%s depth=%s trees=%s lr=%.3f",
        tree_method.upper(),
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

    model = SafeXGBClassifier(**params)

    model.set_params(
        early_stopping_rounds=75
    )

    return model


###################################################
# BUILD FINAL MODEL
###################################################

def build_final_xgboost_model(y):
    pos_weight = compute_class_weight(y)
    params = _base_params(pos_weight)

    params.update({
        "n_estimators": 1400,
        "learning_rate": 0.04,
        "max_depth": 8,
        "reg_alpha": 0.6,
        "reg_lambda": 1.2,
    })

    logger.info("Building final production model.")

    return SafeXGBClassifier(**params)
