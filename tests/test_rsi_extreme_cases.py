import pandas as pd
from core.indicators.technical_indicators import TechnicalIndicators


def test_rsi_flat_market_equals_50():

    df = pd.DataFrame({
        "Close": [100] * 50
    })

    rsi = TechnicalIndicators.rsi(df)

    assert (rsi.dropna() == 50).all()
