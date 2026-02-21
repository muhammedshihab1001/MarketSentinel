import pandas as pd
import numpy as np

from training.backtesting.backtest_engine import (
    BacktestEngine,
    backtest_strategy,
    signal_hit_rate
)


############################################################
# HELPER
############################################################

def assert_no_nan_dict(d):
    for v in d.values():
        if isinstance(v, float):
            assert not np.isnan(v)


############################################################
# CORE ENGINE EXECUTION
############################################################

def test_engine_runs_and_returns_valid_structure():

    prices = np.array([100, 101, 102, 103, 104])
    signals = ["BUY", "HOLD", "SELL", "HOLD", "BUY"]

    engine = BacktestEngine()

    result = engine.run(prices, signals)

    assert isinstance(result, dict)

    required_keys = [
        "final_portfolio",
        "buy_hold_return",
        "trade_count",
        "equity_curve"
    ]

    for key in required_keys:
        assert key in result

    assert result["final_portfolio"] > 0
    assert len(result["equity_curve"]) == len(prices)
    assert_no_nan_dict(result)


############################################################
# CAPITAL CONSERVATION
############################################################

def test_no_negative_equity():

    prices = np.array([100, 90, 80, 70])
    signals = ["BUY", "HOLD", "HOLD", "HOLD"]

    engine = BacktestEngine()

    result = engine.run(prices, signals)

    assert result["final_portfolio"] >= 0
    assert all(v >= 0 for v in result["equity_curve"])


############################################################
# BUY & HOLD CORRECTNESS
############################################################

def test_buy_hold_matches_manual():

    prices = np.array([100, 110])
    signals = ["HOLD", "HOLD"]

    engine = BacktestEngine()

    result = engine.run(prices, signals)

    expected = 110 / 100 - 1

    assert abs(result["buy_hold_return"] - expected) < 1e-8


############################################################
# TRADE COUNT LOGIC
############################################################

def test_trade_count_matches_signal_changes():

    prices = np.array([100, 101, 102, 103])
    signals = ["BUY", "SELL", "BUY", "SELL"]

    engine = BacktestEngine()

    result = engine.run(prices, signals)

    assert result["trade_count"] == 4


############################################################
# WRAPPER FUNCTION CONTRACT
############################################################

def test_wrapper_outputs_valid_metrics():

    df = pd.DataFrame({
        "close": [100, 101, 102, 103],
        "signal": ["BUY", "SELL", "HOLD", "BUY"]
    })

    metrics = backtest_strategy(df)
    hit = signal_hit_rate(df)

    assert isinstance(metrics, dict)
    assert "total_return" in metrics

    assert isinstance(hit, dict)
    assert 0.0 <= hit["hit_rate"] <= 1.0


############################################################
# HIT RATE EDGE CASE
############################################################

def test_hit_rate_edge_case():

    df = pd.DataFrame({
        "close": [100, 100, 100],
        "signal": ["HOLD", "HOLD", "HOLD"]
    })

    hit = signal_hit_rate(df)

    assert 0.0 <= hit["hit_rate"] <= 1.0