from xgboost import XGBClassifier
import numpy as np
import xgboost as xgb
import threading
import logging

from core.schema.feature_schema import FEATURE_COUNT


logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42

MAX_CLASS_WEIGHT = 30.0
MIN_CLASS_WEIGHT = 1.0


###################################################
# 🔥 REAL GPU DETECTION
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

            # THIS is the correct GPU test
            xgb.XGBClassifier(
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                max_depth=1,
                n_estimators=1,
                verbosity=0
            ).fit(
                np.random.rand(20, 4),
                np.random.randint(0, 2, 20)
            )

            logger.info("GPU detected — using CUDA acceleration.")
            _GPU_AVAILABLE = True

        except Exception:

            logger.info("GPU not available — using CPU.")
            _GPU_AVAILABLE = False

        return _GPU_AVAILABLE


def _tree_method():
    return "gpu_hist" if _gpu_verified() else "hist"


def _predictor():
    return "gpu_predictor" if _gpu_verified() else "auto"


###################################################
# CLASS WEIGHT
###################################################

def compute_class_weight(y):

    y = np.asarray(y)

    if len(y) == 0:
        raise RuntimeError("Empty labels.")

    if not np.isfinite(y).all():
        raise RuntimeError("Non-finite labels.")

    pos = float(np.sum(y))
    neg = float(len(y) - pos)

    if pos == 0 or neg == 0:
        raise RuntimeError("Label collapse detected.")

    weight = neg / pos

    weight = float(np.clip(weight, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT))

    logger.info("Computed class weight → %.3f", weight)

    return weight


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
# BASE PARAMS
###################################################

def _base_params(pos_weight, overrides=None):

    params = dict(

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

        tree_method=_tree_method(),
        predictor=_predictor(),

        scale_pos_weight=pos_weight,

        max_delta_step=1,
        grow_policy="depthwise",
        max_bin=192,

        use_label_encoder=False,
        verbosity=0,
    )

    if overrides:
        params.update(overrides)

    logger.info(
        "XGBoost params | device=%s estimators=%s lr=%.3f",
        "GPU" if _gpu_verified() else "CPU",
        params.get("n_estimators"),
        params.get("learning_rate"),
    )

    return params


###################################################
# SAFE WRAPPER
###################################################

class SafeXGBClassifier(XGBClassifier):

    def fit(self, X, y, **kwargs):

        X = np.asarray(X)

        _validate_features(X)

        logger.info("XGBoost training started | rows=%s", len(X))

        # 🔥 Automatic early stopping split
        split = int(len(X) * 0.90)

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        return super().fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=75,
            verbose=False
        )


###################################################
# TRAIN MODEL
###################################################

def build_xgboost_model(y, **overrides):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight, overrides)

    params.setdefault("n_estimators", 600)

    return SafeXGBClassifier(**params)


###################################################
# FINAL MODEL
###################################################

def build_final_xgboost_model(y, **overrides):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight, overrides)

    params.update({
        "n_estimators": 900,
        "learning_rate": 0.025,
        "gamma": 0.20,
        "reg_alpha": 0.9,
        "reg_lambda": 1.4,
    })

    return SafeXGBClassifier(**params)
