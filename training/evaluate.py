"""
MarketSentinel Research Evaluation Module (Ranking-Aligned)
Hybrid-Ready Cross-Sectional Diagnostics
"""

import numpy as np
import pandas as pd


EPSILON = 1e-12
MIN_CS_SIZE = 5


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

    # -----------------------------------------------------
    # Drop NaNs / Infs early
    # -----------------------------------------------------

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if df.empty:
        return {
            "information_coefficient": 0.0,
            "long_short_spread": 0.0,
            "sharpe": 0.0,
            "decile_spread": 0.0,
            "score_std": 0.0,
            "return_std": 0.0,
            "sign_hit_rate": 0.0,
            "num_samples": 0,
            "num_dates": 0,
        }

    grouped = df.groupby("date")

    ic_list = []
    long_short_daily = []
    decile_spreads = []

    # =====================================================
    # CROSS-SECTION LOOP
    # =====================================================

    for date, group in grouped:

        if len(group) < MIN_CS_SIZE:
            continue

        if group["forward_return"].nunique() >= 2:

            corr = group["score"].corr(
                group["forward_return"],
                method="spearman"
            )

            if np.isfinite(corr):
                ic_list.append(corr)

        if len(group) >= 10:

            group = group.sort_values("score")

            quint = max(1, len(group) // 5)
            dec = max(1, len(group) // 10)

            bottom_q = group.head(quint)
            top_q = group.tail(quint)

            spread_q = (
                top_q["forward_return"].mean()
                - bottom_q["forward_return"].mean()
            )

            long_short_daily.append(spread_q)

            bottom_d = group.head(dec)
            top_d = group.tail(dec)

            spread_d = (
                top_d["forward_return"].mean()
                - bottom_d["forward_return"].mean()
            )

            decile_spreads.append(spread_d)

    # =====================================================
    # METRICS
    # =====================================================

    information_coefficient = float(np.mean(ic_list)) if ic_list else 0.0

    if long_short_daily:

        spread_series = pd.Series(long_short_daily)

        long_short_spread = float(spread_series.mean())

        std = spread_series.std()

        if std > EPSILON:

            sharpe = float(
                (spread_series.mean() / std)
                * np.sqrt(252 / 5)
            )

        else:
            sharpe = 0.0

    else:

        long_short_spread = 0.0
        sharpe = 0.0

    decile_spread = float(np.mean(decile_spreads)) if decile_spreads else 0.0

    score_std = float(np.std(df["score"]))
    return_std = float(np.std(df["forward_return"]))

    # =====================================================
    # HIT RATE
    # =====================================================

    valid = df["forward_return"] != 0

    if valid.any():

        sign_hit_rate = float(
            np.mean(
                np.sign(df.loc[valid, "score"])
                == np.sign(df.loc[valid, "forward_return"])
            )
        )

    else:

        sign_hit_rate = 0.0

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