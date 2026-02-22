import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

# =========================================================
# GOVERNANCE THRESHOLDS (Backward Compatibility for Tests)
# =========================================================

XGB_MIN_ACCURACY = 0.50
XGB_MIN_BALANCED_ACCURACY = 0.50
XGB_MIN_ROC_AUC = 0.50

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
    Cross-sectional evaluation for alpha model.
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

        # =================================================
        # VOLATILITY-NORMALIZED SHARPE (INSTITUTIONAL)
        # =================================================

        if spread_series.std() > 0:
            vol_target = 0.02  # 2% daily volatility target
            scaled = spread_series * (vol_target / spread_series.std())
            sharpe = scaled.mean() / scaled.std()
        else:
            sharpe = 0.0

    # -----------------------------------------------------
    # Information Coefficient (rank correlation)
    # -----------------------------------------------------

    ic_list = []

    for date, group in df.groupby("date"):

        if group["forward_return"].nunique() < 2:
            continue

        corr = group["prob"].corr(group["forward_return"], method="spearman")

        if not np.isnan(corr):
            ic_list.append(corr)

    information_coefficient = float(np.mean(ic_list)) if ic_list else 0.0

    # -----------------------------------------------------
    # Hit Rate
    # -----------------------------------------------------

    hit_rate = float(np.mean((y_true == y_pred)))

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc": float(roc_auc),
        "long_short_spread": float(long_short_spread),
        "sharpe": float(sharpe),
        "information_coefficient": float(information_coefficient),
        "hit_rate": float(hit_rate),
    }

    return metrics