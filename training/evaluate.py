import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =========================================================
# GOVERNANCE THRESHOLDS
# =========================================================

XGB_MIN_ACCURACY = 0.50
XGB_MIN_BALANCED_ACCURACY = 0.50
XGB_MIN_ROC_AUC = 0.50
XGB_MIN_F1 = 0.50

XGB_MIN_AUC = XGB_MIN_ROC_AUC


# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================

def evaluate_xgboost(
    y_true,
    y_pred,
    y_prob,
    forward_returns,
    dates,
    enforce_thresholds: bool = True,
):
    """
    Institutional cross-sectional alpha evaluation.

    Includes:
    - Classification metrics
    - Long/short spread performance
    - Vol-normalized Sharpe
    - Information Coefficient (Spearman)
    - Precision / Recall / F1
    - Confusion matrix diagnostics
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    forward_returns = np.asarray(forward_returns)
    dates = np.asarray(dates)

    # -----------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # -----------------------------------------------------
    # Cross-sectional long/short performance
    # -----------------------------------------------------

    df = pd.DataFrame(
        {
            "date": dates,
            "pred": y_pred,
            "forward_return": forward_returns,
            "prob": y_prob,
        }
    )

    long_short_daily = []

    for date, group in df.groupby("date"):

        longs = group[group["pred"] == 1]
        shorts = group[group["pred"] == 0]

        if len(longs) == 0 or len(shorts) == 0:
            continue

        long_ret = longs["forward_return"].mean()
        short_ret = shorts["forward_return"].mean()

        spread = long_ret - short_ret
        long_short_daily.append(spread)

    if len(long_short_daily) == 0:
        sharpe = 0.0
        long_short_spread = 0.0
    else:
        spread_series = pd.Series(long_short_daily)
        long_short_spread = spread_series.mean()

        if spread_series.std() > 0:
            vol_target = 0.02  # 2% daily vol target
            scaled = spread_series * (vol_target / spread_series.std())
            sharpe = scaled.mean() / scaled.std()
        else:
            sharpe = 0.0

    # -----------------------------------------------------
    # Information Coefficient (Spearman)
    # -----------------------------------------------------

    ic_list = []

    for date, group in df.groupby("date"):

        if group["forward_return"].nunique() < 2:
            continue

        corr = group["prob"].corr(
            group["forward_return"],
            method="spearman"
        )

        if not np.isnan(corr):
            ic_list.append(corr)

    information_coefficient = float(np.mean(ic_list)) if ic_list else 0.0

    # -----------------------------------------------------
    # Decile IC (Top/Bottom Strength)
    # -----------------------------------------------------

    decile_ic_list = []

    for date, group in df.groupby("date"):

        if len(group) < 10:
            continue

        group = group.sort_values("prob")

        bottom = group.head(max(1, len(group) // 10))
        top = group.tail(max(1, len(group) // 10))

        spread = top["forward_return"].mean() - bottom["forward_return"].mean()
        decile_ic_list.append(spread)

    decile_spread = float(np.mean(decile_ic_list)) if decile_ic_list else 0.0

    # -----------------------------------------------------
    # Hit Rate
    # -----------------------------------------------------

    hit_rate = float(np.mean((y_true == y_pred)))

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "long_short_spread": float(long_short_spread),
        "sharpe": float(sharpe),
        "information_coefficient": float(information_coefficient),
        "decile_spread": float(decile_spread),
        "hit_rate": float(hit_rate),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }

    # -----------------------------------------------------
    # Governance Enforcement (Optional)
    # -----------------------------------------------------

    if enforce_thresholds:
        if accuracy < XGB_MIN_ACCURACY:
            raise RuntimeError("Accuracy below governance threshold.")
        if balanced_acc < XGB_MIN_BALANCED_ACCURACY:
            raise RuntimeError("Balanced accuracy below threshold.")
        if roc_auc < XGB_MIN_ROC_AUC:
            raise RuntimeError("ROC AUC below threshold.")
        if f1 < XGB_MIN_F1:
            raise RuntimeError("F1 score below threshold.")

    return metrics