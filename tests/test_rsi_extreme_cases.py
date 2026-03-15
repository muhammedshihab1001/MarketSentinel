import numpy as np
import pandas as pd

from core.indicators.technical_indicators import TechnicalIndicators


############################################################
# HELPER
############################################################

def build_df(prices):

    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=len(prices)),
        "close": prices
    })


############################################################
# FLAT MARKET
############################################################

def test_rsi_flat_market_equals_50():

    df = build_df([100.0] * 100)

    rsi = TechnicalIndicators.rsi(df, period=14).dropna()

    # Flat market RSI should stay within valid bounds
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()

    # Early RSI values should be near neutral
    assert np.isclose(rsi.iloc[:5].mean(), 50.0, atol=5)


############################################################
# STRONG UP TREND
############################################################

def test_rsi_strong_uptrend():

    df = build_df(np.linspace(100, 200, 100))

    rsi = TechnicalIndicators.rsi(df, period=14).dropna()

    # Strong uptrend should keep RSI mostly high
    assert (rsi > 70).mean() > 0.8


############################################################
# STRONG DOWN TREND
############################################################

def test_rsi_strong_downtrend():

    df = build_df(np.linspace(200, 100, 100))

    rsi = TechnicalIndicators.rsi(df, period=14).dropna()

    # Strong downtrend should keep RSI mostly low
    assert (rsi < 30).mean() > 0.8


############################################################
# RSI BOUNDS SAFETY
############################################################

def test_rsi_bounds():

    rng = np.random.default_rng(42)

    df = build_df(rng.normal(100, 2, 100))

    rsi = TechnicalIndicators.rsi(df, period=14).dropna()

    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


############################################################
# SHORT SERIES SAFETY
############################################################

def test_rsi_short_series():

    df = build_df(np.linspace(100, 110, 10))

    rsi = TechnicalIndicators.rsi(df, period=14)

    assert len(rsi) == len(df)


############################################################
# OUTPUT TYPE
############################################################

def test_rsi_returns_series():

    df = build_df(np.linspace(100, 110, 50))

    rsi = TechnicalIndicators.rsi(df, period=14)

    assert isinstance(rsi, pd.Series)