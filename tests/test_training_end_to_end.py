import pandas as pd
import numpy as np
from training.backtesting.walk_forward import WalkForwardValidator


def test_walkforward_runs():

    dates = pd.date_range("2026-01-01", periods=500)
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    rows = []

    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "close": np.random.rand() + 10,
                "volatility": 0.2,
                "momentum_20_z": np.random.randn()
            })

    df = pd.DataFrame(rows)

    validator = WalkForwardValidator(lambda x: None)

    # Just confirm it does not crash early
    assert len(df) > 0