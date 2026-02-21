import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score
)

###################################################
# PRODUCTION THRESHOLDS
###################################################

XGB_MIN_ACCURACY = 0.50
XGB_MIN_BALANCED_ACCURACY = 0.50
XGB_MIN_ROC_AUC = 0.51

# ---- Test compatibility alias ----
XGB_MIN_AUC = XGB_MIN_ROC_AUC


def _flatten(arr):
    return np.asarray(arr).reshape(-1)


###################################################
# XGBOOST / CLASSIFICATION
###################################################

def evaluate_xgboost(y_true, y_pred, y_prob=None, enforce_thresholds=False):

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    if len(y_true) != len(y_pred):
        raise RuntimeError("Prediction length mismatch.")

    if len(y_true) == 0:
        raise RuntimeError("Empty evaluation arrays.")

    valid_labels = {0, 1}

    if not set(np.unique(y_true)).issubset(valid_labels):
        raise RuntimeError("Invalid classification labels detected.")

    if not set(np.unique(y_pred)).issubset(valid_labels):
        raise RuntimeError("Invalid prediction labels detected.")

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    roc_auc = None

    if y_prob is not None:

        y_prob = _flatten(y_prob)

        if len(y_prob) != len(y_true):
            raise RuntimeError("Probability length mismatch.")

        if np.any(~np.isfinite(y_prob)):
            raise RuntimeError("Non-finite probabilities detected.")

        if np.any(y_prob < 0) or np.any(y_prob > 1):
            raise RuntimeError("Invalid probabilities detected.")

        if np.std(y_prob) < 1e-8:
            raise RuntimeError("Probability distribution collapsed.")

        if len(np.unique(y_true)) < 2:
            roc_auc = 0.5
        else:
            roc_auc = float(roc_auc_score(y_true, y_prob))

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc
    }

    if enforce_thresholds:
        if accuracy < XGB_MIN_ACCURACY:
            raise RuntimeError("Accuracy below minimum threshold.")

        if balanced_acc < XGB_MIN_BALANCED_ACCURACY:
            raise RuntimeError("Balanced accuracy below minimum threshold.")

        if roc_auc is not None and roc_auc < XGB_MIN_ROC_AUC:
            raise RuntimeError("ROC AUC below minimum threshold.")

    return metrics