"""
MarketSentinel Research Evaluation Module (Ranking-Aligned)
Hybrid-Ready Cross-Sectional Diagnostics
"""

import numpy as np
import pandas as pd


# =========================================================
# MAIN EVALUATION FUNCTION (RANKING-ALIGNED)
# =========================================================

def evaluate_xgboost(
    scores,
    forward_returns,
    dates,
):
    """
    Cross-sectional ranking evaluation.

    Designed for:
    - Research diagnostics
    - IC monitoring
    - Decile spread tracking
    - Alpha strength inspection

    Not used for CI governance.
    """

    scores = np.asarray(scores, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    dates = np.asarray(dates)

    if len(scores) != len(forward_returns):
        raise RuntimeError("Score/return length mismatch.")

    df = pd.DataFrame(
        {
            "date": dates,
            "score": scores,
            "forward_return": forward_returns,
        }
    )

    # =====================================================
    # INFORMATION COEFFICIENT (Spearman)
    # =====================================================

    ic_list = []

    for date, group in df.groupby("date"):

        if group["forward_return"].nunique() < 2:
            continue

        corr = group["score"].corr(
            group["forward_return"],
            method="spearman"
        )

        if not np.isnan(corr):
            ic_list.append(corr)

    information_coefficient = float(np.mean(ic_list)) if ic_list else 0.0

    # =====================================================
    # LONG-SHORT SPREAD (TOP/BOTTOM QUINTILE)
    # =====================================================

    long_short_daily = []

    for date, group in df.groupby("date"):

        if len(group) < 10:
            continue

        group = group.sort_values("score")

        bottom = group.head(max(1, len(group) // 5))
        top = group.tail(max(1, len(group) // 5))

        long_ret = top["forward_return"].mean()
        short_ret = bottom["forward_return"].mean()

        spread = long_ret - short_ret
        long_short_daily.append(spread)

    if long_short_daily:
        spread_series = pd.Series(long_short_daily)

        long_short_spread = float(spread_series.mean())

        if spread_series.std() > 0:
            sharpe = float(
                (spread_series.mean() / spread_series.std())
                * np.sqrt(252 / 5)
            )
        else:
            sharpe = 0.0
    else:
        long_short_spread = 0.0
        sharpe = 0.0

    # =====================================================
    # DECILE SPREAD
    # =====================================================

    decile_spreads = []

    for date, group in df.groupby("date"):

        if len(group) < 10:
            continue

        group = group.sort_values("score")

        bottom = group.head(max(1, len(group) // 10))
        top = group.tail(max(1, len(group) // 10))

        spread = top["forward_return"].mean() - bottom["forward_return"].mean()
        decile_spreads.append(spread)

    decile_spread = float(np.mean(decile_spreads)) if decile_spreads else 0.0

    # =====================================================
    # RANK CORRELATION STABILITY
    # =====================================================

    score_std = float(np.std(scores))
    return_std = float(np.std(forward_returns))

    # =====================================================
    # HIT RATE (SIGN CONSISTENCY)
    # =====================================================

    sign_hit_rate = float(
        np.mean(
            np.sign(scores) == np.sign(forward_returns)
        )
    )

    metrics = {
        "information_coefficient": information_coefficient,
        "long_short_spread": long_short_spread,
        "sharpe": sharpe,
        "decile_spread": decile_spread,
        "score_std": score_std,
        "return_std": return_std,
        "sign_hit_rate": sign_hit_rate,
        "num_samples": int(len(df)),
        "num_dates": int(df["date"].nunique()),
    }

    return metrics