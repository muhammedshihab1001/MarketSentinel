import pandas as pd
from training.backtesting.backtest_engine import backtest_strategy, signal_hit_rate


def sample_backtest_df():
    """
    Create a small but realistic dataset for backtesting.
    """
    data = {
        "close": [
            100, 101, 102, 101, 103,
            104, 103, 105, 106, 107
        ],
        "signal": [
            "BUY", "HOLD", "BUY", "SELL", "BUY",
            "HOLD", "SELL", "BUY", "BUY", "HOLD"
        ]
    }
    return pd.DataFrame(data)


def test_strategy_performance_thresholds():
    df = sample_backtest_df()

    metrics = backtest_strategy(df)
    signal_metrics = signal_hit_rate(df)

    # ---- Quality Gates ----
    assert metrics["sharpe_ratio"] >= 0.8, "Sharpe ratio too low"
    assert signal_metrics["hit_rate"] >= 0.50, "Signal hit rate too low"
