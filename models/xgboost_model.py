from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import logging
import threading

from core.schema.feature_schema import FEATURE_COUNT

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42
MIN_PROB_STD = 1e-5


###################################################
# GPU DETECTION
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
                {"tree_method": "gpu_hist", "max_depth": 1, "verbosity": 0},
                dtrain,
                num_boost_round=1
            )

            _GPU_AVAILABLE = True
            logger.info("CUDA backend verified.")

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
    weight = float(np.clip(weight, 0.9, 5.0))

    logger.info("Computed class weight = %.3f", weight)

    return weight


###################################################
# SAFE CLASSIFIER
###################################################

class SafeXGBClassifier(XGBClassifier):

    def fit(self, X, y, **kwargs):

        if not hasattr(X, "shape"):
            raise RuntimeError("Invalid feature matrix.")

        if X.shape[1] != FEATURE_COUNT:
            raise RuntimeError(
                f"Feature schema mismatch. Expected {FEATURE_COUNT}, got {X.shape[1]}"
            )

        arr = X.to_numpy() if hasattr(X, "to_numpy") else X

        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite feature values detected.")

        logger.info(
            "XGBoost training | rows=%s cols=%s",
            X.shape[0],
            X.shape[1]
        )

        model = super().fit(X, y, **kwargs)

        preds = model.predict_proba(X)[:, 1]

        mean = float(np.mean(preds))
        std = float(np.std(preds))

        logger.info(
            "Train prob stats | mean=%.4f std=%.4f min=%.4f max=%.4f",
            mean,
            std,
            float(np.min(preds)),
            float(np.max(preds))
        )

        if std < MIN_PROB_STD:
            raise RuntimeError("Probability collapse detected after training.")

        return model


###################################################
# PARAM BUILDER (REDUCED CAPACITY VERSION)
###################################################

def _base_params(pos_weight):

    params = dict(

        # ↓↓↓ MUCH LOWER CAPACITY
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,

        # Stronger regularization
        subsample=0.65,
        colsample_bytree=0.6,
        colsample_bylevel=0.6,

        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.8,
        reg_lambda=2.5,

        max_bin=256,

        tree_method=_tree_method(),
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
        scale_pos_weight=pos_weight,
        verbosity=0
    )

    logger.info(
        "XGBoost params | depth=%s trees=%s lr=%.3f",
        params["max_depth"],
        params["n_estimators"],
        params["learning_rate"]
    )

    return params


###################################################
# BUILD MODEL
###################################################

def build_xgboost_pipeline(y):

    pos_weight = compute_class_weight(y)
    params = _base_params(pos_weight)

    model = SafeXGBClassifier(**params)

    logger.info("XGBoost model built successfully (regularized).")

    return model