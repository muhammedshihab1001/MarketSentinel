import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score
)

# =========================================================
# GOVERNANCE THRESHOLDS
# =========================================================

# For financial alpha modeling:
# 0.50 = random → NOT acceptable
# Model must be strictly better than random.

XGB_MIN_ACCURACY = 0.50
XGB_MIN_BALANCED_ACCURACY = 0.50
XGB_MIN_ROC_AUC = 0.50

# Backward compatibility
XGB_MIN_AUC = XGB_MIN_ROC_AUC


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _flatten(arr):
    return np.asarray(arr).reshape(-1)


# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================

def evaluate_xgboost(
    y_true,
    y_pred,
    y_prob=None,
    forward_returns=None,
    dates=None,
    enforce_thresholds=False
):

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    if len(y_true) != len(y_pred):
        raise RuntimeError("Prediction length mismatch.")

    if len(y_true) == 0:
        raise RuntimeError("Empty evaluation arrays.")

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise RuntimeError("Invalid classification labels detected.")

    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise RuntimeError("Invalid prediction labels detected.")

    # =====================================================
    # CLASSIFICATION METRICS
    # =====================================================

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    roc_auc = None

    if y_prob is not None:

        y_prob = _flatten(y_prob)

        if len(y_prob) != len(y_true):
            raise RuntimeError("Probability length mismatch.")

        if np.any(~np.isfinite(y_prob)):
            raise RuntimeError("Non-finite probabilities detected.")

        if np.any((y_prob < 0) | (y_prob > 1)):
            raise RuntimeError("Invalid probabilities detected.")

        if np.std(y_prob) < 1e-8:
            raise RuntimeError("Probability distribution collapsed.")

        if len(np.unique(y_true)) < 2:
            roc_auc = 0.5
        else:
            roc_auc = float(roc_auc_score(y_true, y_prob))

    # =====================================================
    # ALPHA METRICS (OPTIONAL)
    # =====================================================

    long_short_spread = None
    sharpe = None
    information_coefficient = None
    hit_rate = None

    if forward_returns is not None:

        forward_returns = _flatten(forward_returns)

        if len(forward_returns) != len(y_true):
            raise RuntimeError("Forward return length mismatch.")

        longs = forward_returns[y_pred == 1]
        shorts = forward_returns[y_pred == 0]

        if len(longs) > 0 and len(shorts) > 0:
            long_short_spread = float(
                np.mean(longs) - np.mean(shorts)
            )

        if np.std(forward_returns) > 0:
            sharpe = float(
                np.mean(forward_returns) /
                np.std(forward_returns)
            )

        if y_prob is not None:
            information_coefficient = float(
                np.corrcoef(y_prob, forward_returns)[0, 1]
            )

        hit_rate = float(np.mean((forward_returns > 0)))

    # =====================================================
    # METRIC DICT
    # =====================================================

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "long_short_spread": long_short_spread,
        "sharpe": sharpe,
        "information_coefficient": information_coefficient,
        "hit_rate": hit_rate,
    }

    # =====================================================
    # STRICT GOVERNANCE ENFORCEMENT
    # =====================================================

    if enforce_thresholds:

        if accuracy < XGB_MIN_ACCURACY:
            raise RuntimeError("Accuracy below minimum threshold.")

        if balanced_acc < XGB_MIN_BALANCED_ACCURACY:
            raise RuntimeError("Balanced accuracy below minimum threshold.")

        # STRICT RULE: must be strictly better than random
        if roc_auc is not None and roc_auc <= XGB_MIN_ROC_AUC:
            raise RuntimeError("ROC AUC not better than random.")

    return metrics