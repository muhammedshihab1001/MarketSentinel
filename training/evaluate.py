import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score
)

# ===============================
# GOVERNANCE THRESHOLDS
# ===============================

XGB_MIN_ACCURACY = 0.48
XGB_MIN_BALANCED_ACCURACY = 0.48
XGB_MIN_ROC_AUC = 0.50
XGB_MIN_SHARPE = 0.10
XGB_MIN_SPREAD = 0.0

# Backward compatibility
XGB_MIN_AUC = XGB_MIN_ROC_AUC


# ===============================
# UTIL
# ===============================

def _flatten(arr):
    return np.asarray(arr).reshape(-1)


def _safe_std(arr):
    s = float(np.std(arr))
    return max(s, 1e-12)


# ===============================
# CORE EVALUATION
# ===============================

def evaluate_xgboost(
    y_true,
    y_pred,
    y_prob=None,
    forward_returns=None,
    dates=None,
    enforce_thresholds=False
):
    """
    Institutional-grade evaluation.

    Supports:
    - classification metrics
    - cross-sectional alpha metrics
    """

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

    # ===============================
    # CLASSIFICATION METRICS
    # ===============================

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

    # ===============================
    # ALPHA METRICS
    # ===============================

    sharpe = None
    spread = None
    ic = None
    hit_rate = None

    if forward_returns is not None and y_prob is not None:

        forward_returns = _flatten(forward_returns)

        if len(forward_returns) != len(y_prob):
            raise RuntimeError("Forward return length mismatch.")

        df = pd.DataFrame({
            "prob": y_prob,
            "ret": forward_returns
        })

        if dates is not None:
            df["date"] = dates
        else:
            df["date"] = 0  # single batch fallback

        daily_spreads = []

        for _, group in df.groupby("date"):

            if len(group) < 10:
                continue

            group = group.sort_values("prob")

            n = len(group)
            bucket = int(n * 0.3)

            if bucket < 1:
                continue

            short_ret = group.iloc[:bucket]["ret"].mean()
            long_ret = group.iloc[-bucket:]["ret"].mean()

            daily_spreads.append(long_ret - short_ret)

        if len(daily_spreads) > 5:

            daily_spreads = np.array(daily_spreads)

            spread = float(np.mean(daily_spreads))
            sharpe = float(
                np.mean(daily_spreads) /
                _safe_std(daily_spreads)
            )

        # Information Coefficient
        if len(df) > 20:
            ic, _ = spearmanr(df["prob"], df["ret"])
            ic = float(ic)

        # Hit rate
        if spread is not None:
            hit_rate = float(np.mean(np.array(daily_spreads) > 0))

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "long_short_spread": spread,
        "sharpe": sharpe,
        "information_coefficient": ic,
        "hit_rate": hit_rate
    }

    # ===============================
    # GOVERNANCE ENFORCEMENT
    # ===============================

    if enforce_thresholds:

        if accuracy < XGB_MIN_ACCURACY:
            raise RuntimeError("Accuracy below minimum threshold.")

        if balanced_acc < XGB_MIN_BALANCED_ACCURACY:
            raise RuntimeError("Balanced accuracy below minimum threshold.")

        if roc_auc is not None and roc_auc < XGB_MIN_ROC_AUC:
            raise RuntimeError("ROC AUC below minimum threshold.")

        if sharpe is not None and sharpe < XGB_MIN_SHARPE:
            raise RuntimeError("Sharpe below governance threshold.")

        if spread is not None and spread < XGB_MIN_SPREAD:
            raise RuntimeError("Long-short spread negative.")

    return metrics