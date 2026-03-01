import numpy as np
import pandas as pd

from core.indicators.technical_indicators import TechnicalIndicators


############################################################
# FLAT MARKET
############################################################

def test_rsi_flat_market_equals_50():

    dates = pd.date_range("2025-01-01", periods=100)

    df = pd.DataFrame({
        "date": dates,
        "close": [100.0] * 100
    })

    rsi = TechnicalIndicators.rsi(df, period=14)

    # Drop initial NaNs from warmup window
    rsi = rsi.dropna()

    # RSI of flat market should be ~50
    assert np.allclose(rsi.values, 50.0, atol=1e-6)


############################################################
# STRONG UP TREND
############################################################

def test_rsi_strong_uptrend():

    dates = pd.date_range("2025-01-01", periods=100)

    df = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 200, 100)
    })

    rsi = TechnicalIndicators.rsi(df, period=14).dropna()

    # Strong uptrend RSI should be high
    assert (rsi > 70).mean() > 0.8


############################################################
# STRONG DOWN TREND
############################################################

def test_rsi_strong_downtrend():

    dates = pd.date_range("2025-01-01", periods=100)

    df = pd.DataFrame({
        "date": dates,
        "close": np.linspace(200, 100, 100)
    })

    rsi = TechnicalIndicators.rsi(df, period=14).dropna()

    # Strong downtrend RSI should be low
    assert (rsi < 30).mean() > 0.8


############################################################
# RSI BOUNDS SAFETY
############################################################

def test_rsi_bounds():

    dates = pd.date_range("2025-01-01", periods=100)

    prices = np.random.normal(100, 2, 100)

    df = pd.DataFrame({
        "date": dates,
        "close": prices
    })

    rsi = TechnicalIndicators.rsi(df, period=14).dropna()

    assert (rsi >= 0).all()
    assert (rsi <= 100).all()