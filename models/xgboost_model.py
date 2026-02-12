from xgboost import XGBClassifier
import numpy as np


SEED = 42


def compute_class_weight(y):

    pos = y.sum()
    neg = len(y) - pos

    if pos == 0:
        return 1.0

    return neg / pos


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

        tree_method="hist",

        n_jobs=1,

        scale_pos_weight=pos_weight,

        reg_alpha=0.5,
        reg_lambda=1.0,

        min_child_weight=2,
        gamma=0.1
    )

    return model


###################################################
# FULL TRAIN (FINAL MODEL)
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

        tree_method="hist",

        n_jobs=1,

        scale_pos_weight=pos_weight,

        reg_alpha=0.7,
        reg_lambda=1.2,

        min_child_weight=2,
        gamma=0.15
    )

    return model
