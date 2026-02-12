from xgboost import XGBClassifier
import numpy as np
import xgboost as xgb


SEED = 42
MAX_CLASS_WEIGHT = 50.0


###################################################
# SAFE GPU DETECTION (PRODUCTION GRADE)
###################################################

def get_device():
    """
    Institutional GPU detection.

    No fake training.
    No private APIs.
    Docker-safe.
    CI-safe.
    """

    try:
        if hasattr(xgb, "cuda_is_available"):
            if xgb.cuda_is_available():
                return "cuda"
    except Exception:
        pass

    return "cpu"


DEVICE = get_device()


###################################################
# CLASS WEIGHT
###################################################

def compute_class_weight(y):

    if y is None or len(y) == 0:
        raise RuntimeError("Empty labels provided to XGBoost.")

    y = np.asarray(y)

    if np.isnan(y).any():
        raise RuntimeError("NaN labels detected.")

    pos = float(np.sum(y))
    neg = float(len(y) - pos)

    if pos == 0 or neg == 0:
        raise RuntimeError(
            "Label collapse detected — model cannot train."
        )

    weight = neg / pos

    return float(min(weight, MAX_CLASS_WEIGHT))


###################################################
# BASE PARAMS
###################################################

def _base_params(pos_weight):

    params = dict(

        # tree structure
        max_depth=5,
        learning_rate=0.03,

        subsample=0.85,
        colsample_bytree=0.85,

        min_child_weight=2,
        gamma=0.1,

        reg_alpha=0.5,
        reg_lambda=1.0,

        # stability
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,

        # modern xgboost
        tree_method="hist",
        device=DEVICE,

        use_label_encoder=False,

        # imbalance
        scale_pos_weight=pos_weight,

        # prevents pathological trees
        max_delta_step=1,

        # memory + stability
        grow_policy="depthwise",
        max_bin=256
    )

    return params


###################################################
# TRAIN MODEL
###################################################

def build_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight)

    params["n_estimators"] = 650

    return XGBClassifier(**params)


###################################################
# FINAL MODEL
###################################################

def build_final_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight)

    params.update({
        "n_estimators": 850,
        "learning_rate": 0.025,
        "reg_alpha": 0.7,
        "reg_lambda": 1.2,
        "gamma": 0.15
    })

    return XGBClassifier(**params)
