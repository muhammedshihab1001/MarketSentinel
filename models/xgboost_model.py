from xgboost import XGBClassifier
import numpy as np
import logging

from core.schema.feature_schema import FEATURE_COUNT


logger = logging.getLogger("marketsentinel.xgboost")

SEED = 42


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
# FEATURE SAFETY
###################################################

def _validate_features(X):

    if X.shape[1] != FEATURE_COUNT:
        raise RuntimeError(
            f"Feature schema mismatch. Expected {FEATURE_COUNT}, got {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite feature values detected.")


###################################################
# SAFE CLASSIFIER
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

        # Remove unsupported kwargs (XGBoost 2.x safety)
        kwargs.pop("early_stopping_rounds", None)
        kwargs.pop("eval_set", None)

        return super().fit(X, y, **kwargs)


###################################################
# PARAM BUILDER (STRICT CPU SAFE)
###################################################

def _base_params(pos_weight):

    params = dict(

        # CORE
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,

        # REGULARIZATION
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.15,
        reg_alpha=0.7,
        reg_lambda=1.2,

        # SAFETY
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,

        # FORCE CPU SAFE
        tree_method="hist",

        # IMBALANCE
        scale_pos_weight=pos_weight,

        verbosity=0
    )

    logger.info(
        "XGBoost params | CPU hist | lr=%.3f",
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

    logger.info("Building final production model.")

    return SafeXGBClassifier(**params)
