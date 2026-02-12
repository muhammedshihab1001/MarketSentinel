from xgboost import XGBClassifier
import numpy as np
import xgboost as xgb


SEED = 42

MAX_CLASS_WEIGHT = 50.0


###################################################
# GPU DETECTION
###################################################

def get_tree_method():

    try:
        if xgb.core._has_cuda_support():
            return "gpu_hist"
    except Exception:
        pass

    return "hist"


TREE_METHOD = get_tree_method()


###################################################
# CLASS WEIGHT
###################################################

def compute_class_weight(y):

    if y is None or len(y) == 0:
        raise RuntimeError("Empty labels provided to XGBoost.")

    if np.isnan(y).any():
        raise RuntimeError("NaN labels detected.")

    pos = float(np.sum(y))
    neg = float(len(y) - pos)

    if pos == 0 or neg == 0:
        raise RuntimeError(
            "Label collapse detected — model cannot train."
        )

    weight = neg / pos

    return min(weight, MAX_CLASS_WEIGHT)


###################################################
# TRAIN MODEL
###################################################

def build_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    model = XGBClassifier(

        n_estimators=700,
        max_depth=5,
        learning_rate=0.03,

        subsample=0.85,
        colsample_bytree=0.85,

        eval_metric="logloss",

        random_state=SEED,

        tree_method=TREE_METHOD,

        n_jobs=-1,

        scale_pos_weight=pos_weight,

        reg_alpha=0.5,
        reg_lambda=1.0,

        min_child_weight=2,
        gamma=0.1
    )

    return model


###################################################
# FINAL MODEL
###################################################

def build_final_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    model = XGBClassifier(

        n_estimators=900,
        max_depth=5,
        learning_rate=0.025,

        subsample=0.85,
        colsample_bytree=0.85,

        eval_metric="logloss",

        random_state=SEED,

        tree_method=TREE_METHOD,

        n_jobs=-1,

        scale_pos_weight=pos_weight,

        reg_alpha=0.7,
        reg_lambda=1.2,

        min_child_weight=2,
        gamma=0.15
    )

    return model
