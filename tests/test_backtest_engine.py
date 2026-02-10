import pandas as pd
import numpy as np

from training.backtesting.backtest_engine import (
    BacktestEngine,
    backtest_strategy,
    signal_hit_rate
)


# ---------------------------------------------------
# CORE ENGINE SAFETY
# ---------------------------------------------------

def test_engine_runs_without_crashing():

    prices = np.array([100, 101, 102, 103])
    signals = ["BUY", "HOLD", "SELL", "HOLD"]

    engine = BacktestEngine()

    result = engine.run(prices, signals)

    assert "final_portfolio" in result
    assert result["final_portfolio"] > 0


# ---------------------------------------------------
# CAPITAL CONSERVATION
# ---------------------------------------------------

def test_no_negative_portfolio():

    prices = np.array([100, 99, 98, 97])
    signals = ["BUY", "HOLD", "HOLD", "HOLD"]

    engine = BacktestEngine()

    result = engine.run(prices, signals)

    assert result["final_portfolio"] >= 0


# ---------------------------------------------------
# BUY HOLD BASELINE
# ---------------------------------------------------

def test_buy_hold_matches_manual():

    prices = np.array([100, 110])
    signals = ["HOLD", "HOLD"]

    engine = BacktestEngine()

    result = engine.run(prices, signals)

    expected = 110 / 100 - 1

    assert abs(result["buy_hold_return"] - expected) < 1e-6


# ---------------------------------------------------
# TRADE COUNT
# ---------------------------------------------------

def test_trade_count_increments():

    prices = np.array([100, 101, 102, 103])
    signals = ["BUY", "SELL", "BUY", "SELL"]

    engine = BacktestEngine()

    result = engine.run(prices, signals)

    assert result["trade_count"] == 4


# ---------------------------------------------------
# WRAPPER COMPATIBILITY
# ---------------------------------------------------

def test_wrapper_functions():

    df = pd.DataFrame({
        "close": [100, 101, 102],
        "signal": ["BUY", "SELL", "HOLD"]
    })

    metrics = backtest_strategy(df)
    hit = signal_hit_rate(df)

    assert "total_return" in metrics
    assert 0 <= hit["hit_rate"] <= 1
