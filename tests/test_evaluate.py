"""
Tests for training/evaluate.py — evaluate_xgboost()
Cross-sectional ranking evaluation diagnostics.
"""

import numpy as np
import pandas as pd
import pytest

from training.evaluate import evaluate_xgboost


############################################################
# HELPER
############################################################

def _build_eval_data(n_dates=50, n_tickers=20, seed=42):
    """Build synthetic scores/returns/dates for evaluation."""
    rng = np.random.default_rng(seed)

    dates = np.repeat(
        pd.date_range("2024-01-01", periods=n_dates).values,
        n_tickers,
    )
    scores = rng.normal(0, 1, n_dates * n_tickers).astype(float)
    forward_returns = rng.normal(0, 0.02, n_dates * n_tickers).astype(float)

    return scores, forward_returns, dates


############################################################
# BASIC EXECUTION
############################################################

def test_evaluate_runs_without_crashing():

    scores, fwd, dates = _build_eval_data()

    result = evaluate_xgboost(scores, fwd, dates)

    assert isinstance(result, dict)


############################################################
# REQUIRED KEYS
############################################################

def test_evaluate_returns_required_keys():

    scores, fwd, dates = _build_eval_data()

    result = evaluate_xgboost(scores, fwd, dates)

    required = {
        "information_coefficient",
        "long_short_spread",
        "sharpe",
        "decile_spread",
        "score_std",
        "return_std",
        "sign_hit_rate",
        "num_samples",
        "num_dates",
    }

    assert required.issubset(result.keys())


############################################################
# METRIC FINITENESS
############################################################

def test_evaluate_metrics_are_finite():

    scores, fwd, dates = _build_eval_data()

    result = evaluate_xgboost(scores, fwd, dates)

    for key, value in result.items():
        assert np.isfinite(value), f"Non-finite value for {key}: {value}"


############################################################
# IC RANGE
############################################################

def test_ic_in_valid_range():

    scores, fwd, dates = _build_eval_data()

    result = evaluate_xgboost(scores, fwd, dates)

    # Spearman IC must be in [-1, 1]
    assert -1.0 <= result["information_coefficient"] <= 1.0


############################################################
# HIT RATE RANGE
############################################################

def test_hit_rate_in_valid_range():

    scores, fwd, dates = _build_eval_data()

    result = evaluate_xgboost(scores, fwd, dates)

    assert 0.0 <= result["sign_hit_rate"] <= 1.0


############################################################
# PERFECT POSITIVE SIGNAL
############################################################

def test_perfect_signal_has_positive_ic():
    """When scores perfectly predict returns, IC should be positive."""
    rng = np.random.default_rng(99)

    n_dates, n_tickers = 30, 20
    dates = np.repeat(
        pd.date_range("2024-01-01", periods=n_dates).values,
        n_tickers,
    )
    # Scores = forward returns (perfect signal)
    forward_returns = rng.normal(0, 0.02, n_dates * n_tickers)
    scores = forward_returns + rng.normal(0, 0.001, n_dates * n_tickers)

    result = evaluate_xgboost(scores, forward_returns, dates)

    assert result["information_coefficient"] > 0.5
    assert result["long_short_spread"] > 0


############################################################
# INVERTED SIGNAL
############################################################

def test_inverted_signal_has_negative_ic():
    """When scores are negatively correlated with returns, IC < 0."""
    rng = np.random.default_rng(77)

    n_dates, n_tickers = 30, 20
    dates = np.repeat(
        pd.date_range("2024-01-01", periods=n_dates).values,
        n_tickers,
    )
    forward_returns = rng.normal(0, 0.02, n_dates * n_tickers)
    scores = -forward_returns + rng.normal(0, 0.001, n_dates * n_tickers)

    result = evaluate_xgboost(scores, forward_returns, dates)

    assert result["information_coefficient"] < -0.5


############################################################
# LENGTH MISMATCH RAISES
############################################################

def test_length_mismatch_raises():

    with pytest.raises(RuntimeError, match="mismatch"):
        evaluate_xgboost(
            scores=np.array([1.0, 2.0]),
            forward_returns=np.array([1.0]),
            dates=np.array(["2024-01-01"]),
        )


############################################################
# SAMPLE COUNT CORRECT
############################################################

def test_sample_and_date_counts():

    scores, fwd, dates = _build_eval_data(n_dates=40, n_tickers=15)

    result = evaluate_xgboost(scores, fwd, dates)

    assert result["num_samples"] == 40 * 15
    assert result["num_dates"] == 40


############################################################
# DETERMINISM
############################################################

def test_evaluate_is_deterministic():

    scores, fwd, dates = _build_eval_data()

    r1 = evaluate_xgboost(scores, fwd, dates)
    r2 = evaluate_xgboost(scores, fwd, dates)

    for key in r1:
        assert r1[key] == r2[key], f"Non-deterministic: {key}"
