from xgboost import XGBClassifier
import numpy as np
import xgboost as xgb


SEED = 42
MAX_CLASS_WEIGHT = 50.0


###################################################
# SAFE GPU DETECTION (Institutional)
###################################################

def get_tree_method():
    """
    Production-safe GPU detection.

    Never uses private APIs.
    Works across:
    ✔ docker
    ✔ CI
    ✔ Windows
    ✔ Linux
    """

    try:
        # attempt tiny GPU training
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, 10)

        test = XGBClassifier(
            tree_method="gpu_hist",
            n_estimators=1,
            max_depth=1
        )

        test.fit(X, y)

        return "gpu_hist"

    except Exception:
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
# BASE PARAMS (shared)
###################################################

def _base_params(pos_weight):

    return dict(

        # trees
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

        # CRITICAL
        tree_method=TREE_METHOD,
        predictor="gpu_predictor" if TREE_METHOD == "gpu_hist" else "auto",

        # histogram quality
        max_bin=256,
        deterministic_histogram=True,

        # modern xgboost
        use_label_encoder=False,

        # imbalance
        scale_pos_weight=pos_weight
    )


###################################################
# TRAIN MODEL
###################################################

def build_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight)

    params["n_estimators"] = 700

    return XGBClassifier(**params)


###################################################
# FINAL MODEL
###################################################

def build_final_xgboost_model(y):

    pos_weight = compute_class_weight(y)

    params = _base_params(pos_weight)

    params.update({
        "n_estimators": 900,
        "learning_rate": 0.025,
        "reg_alpha": 0.7,
        "reg_lambda": 1.2,
        "gamma": 0.15
    })

    return XGBClassifier(**params)
