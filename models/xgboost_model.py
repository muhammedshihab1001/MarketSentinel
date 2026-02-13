from xgboost import XGBClassifier
import numpy as np
import xgboost as xgb

from core.schema.feature_schema import FEATURE_COUNT


SEED = 42

MAX_CLASS_WEIGHT = 30.0
MIN_CLASS_WEIGHT = 1.0


###################################################
# HARD GPU VERIFICATION
###################################################

def _gpu_verified():
    """
    Institutional GPU probe.

    Actually trains a tiny booster.
    Prevents fake CUDA environments.
    """

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

        return True

    except Exception:
        return False


DEVICE = "cuda" if _gpu_verified() else "cpu"


###################################################
# CLASS WEIGHT (INSTITUTIONAL SAFE)
###################################################

def compute_class_weight(y):

    if y is None or len(y) == 0:
        raise RuntimeError("Empty labels provided to XGBoost.")

    y = np.asarray(y)

    if not np.isfinite(y).all():
        raise RuntimeError("Non-finite labels detected.")

    pos = float(np.sum(y))
    neg = float(len(y) - pos)

    if pos == 0 or neg == 0:
        raise RuntimeError(
            "Label collapse detected — model cannot train."
        )

    weight = neg / pos

    if not np.isfinite(weight):
        raise RuntimeError("Invalid class weight.")

    return float(
        np.clip(
            weight,
            MIN_CLASS_WEIGHT,
            MAX_CLASS_WEIGHT
        )
    )


###################################################
# FEATURE SAFETY
###################################################

def _validate_feature_count(X):

    if X.shape[1] != FEATURE_COUNT:
        raise RuntimeError(
            f"Feature schema mismatch. Expected {FEATURE_COUNT}, got {X.shape[1]}"
        )


###################################################
# BASE PARAMS
###################################################

def _base_params(pos_weight):

    params = dict(

        ############################
        # TREE STRUCTURE
        ############################

        max_depth=5,
        learning_rate=0.03,

        subsample=0.80,
        colsample_bytree=0.80,

        min_child_weight=3,
        gamma=0.15,

        reg_alpha=0.7,
        reg_lambda=1.2,

        ############################
        # STABILITY
        ############################

        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,                 # deterministic

        tree_method="hist",
        device=DEVICE,

        sampling_method="uniform",

        ############################
        # IMBALANCE
        ############################

        scale_pos_weight=pos_weight,

        ############################
        # PREVENT EXTREME TREES
        ############################

        max_delta_step=1,
        grow_policy="depthwise",
        max_bin=192,

        use_label_encoder=False,
        verbosity=0
    )

    return params


###################################################
# TRAIN MODEL
###################################################

def build_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight)

    params["n_estimators"] = 600

    return XGBClassifier(**params)


###################################################
# FINAL MODEL (CHAMPION TRAIN)
###################################################

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

    return XGBClassifier(**params)
