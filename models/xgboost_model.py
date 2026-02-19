from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import logging
import threading

from core.schema.feature_schema import FEATURE_COUNT

logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42

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

    # More stable clipping
    weight = float(np.clip(weight, 0.8, 10.0))

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

        if hasattr(X, "to_numpy"):
            arr = X.to_numpy()
        else:
            arr = X

        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite feature values detected.")

        logger.info(
            "XGBoost training | rows=%s cols=%s",
            X.shape[0],
            X.shape[1]
        )

        model = super().fit(X, y, **kwargs)

        # 🔥 Log probability dispersion after training
        preds = model.predict_proba(X)[:, 1]
        logger.info(
            "Train prob stats | mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(np.mean(preds)),
            float(np.std(preds)),
            float(np.min(preds)),
            float(np.max(preds))
        )

        return model


###################################################
# PARAM BUILDER (BALANCED ALPHA MODE)
###################################################

def _base_params(pos_weight):

    tree_method = _tree_method()

    params = dict(

        # Trees
        n_estimators=900,
        max_depth=5,
        learning_rate=0.04,

        # Randomization (IMPORTANT for cross-sectional ML)
        subsample=0.85,
        colsample_bytree=0.75,
        colsample_bylevel=0.75,

        # Regularization (reduced to avoid flattening)
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.3,
        reg_lambda=1.2,

        max_bin=256,

        tree_method=tree_method,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
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
# BUILD TRAIN MODEL
###################################################

def build_xgboost_model(y):

    pos_weight = compute_class_weight(y)
    params = _base_params(pos_weight)

    model = SafeXGBClassifier(**params)

    return model


###################################################
# BUILD FINAL MODEL
###################################################

def build_final_xgboost_model(y):

    pos_weight = compute_class_weight(y)
    params = _base_params(pos_weight)

    # Slightly stronger but controlled
    params.update({
        "n_estimators": 1100,
        "learning_rate": 0.035,
        "max_depth": 6,
        "reg_alpha": 0.4,
        "reg_lambda": 1.4,
    })

    logger.info("Building final production model.")

    return SafeXGBClassifier(**params)
