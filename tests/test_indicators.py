import pandas as pd
from training.indicators.technical import TechnicalIndicators


def test_rsi_runs():
    df = pd.DataFrame({
        "Close": [100,101,102,103,104,105,106,107,108,109,110]
    })

    rsi = TechnicalIndicators.rsi(df)

    assert rsi is not None
    assert len(rsi) == len(df)
