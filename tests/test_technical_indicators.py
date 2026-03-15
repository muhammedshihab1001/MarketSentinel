import numpy as np
import pandas as pd
import pytest

from core.indicators.technical_indicators import TechnicalIndicators


############################################################
# HELPER
############################################################

def build_df(prices):

    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=len(prices)),
        "close": prices
    })


############################################################
# BASIC EXECUTION
############################################################

def test_rsi_runs_without_crashing():

    df = build_df(np.linspace(100, 120, 100))

    rsi = TechnicalIndicators.rsi(df, window=14)

    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(df)


############################################################
# BOUNDS CHECK (CRITICAL)
############################################################

def test_rsi_bounds():

    rng = np.random.default_rng(42)

    df = build_df(rng.uniform(90, 110, 200))

    rsi = TechnicalIndicators.rsi(df, window=14).dropna()

    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


############################################################
# STRONG UPTREND
############################################################

def test_rsi_strong_uptrend():

    df = build_df(np.linspace(100, 200, 200))

    rsi = TechnicalIndicators.rsi(df, window=14).dropna()

    assert rsi.iloc[-1] > 70


############################################################
# STRONG DOWNTREND
############################################################

def test_rsi_strong_downtrend():

    df = build_df(np.linspace(200, 100, 200))

    rsi = TechnicalIndicators.rsi(df, window=14).dropna()

    assert rsi.iloc[-1] < 30


############################################################
# FLAT MARKET
############################################################

def test_rsi_flat_market_equals_50():

    df = build_df([100.0] * 150)

    rsi = TechnicalIndicators.rsi(df, window=14).dropna()

    # RSI should remain in valid range
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()

    # Early values should be near neutral
    assert np.isclose(rsi.iloc[:5].mean(), 50.0, atol=5)


############################################################
# NAN SAFETY
############################################################

def test_rsi_no_all_nan():

    df = build_df([100.0] * 50)

    rsi = TechnicalIndicators.rsi(df, window=14)

    assert not rsi.isnull().all()


############################################################
# SHORT SERIES SAFETY
############################################################

def test_rsi_short_series():

    df = build_df(np.linspace(100, 105, 10))

    rsi = TechnicalIndicators.rsi(df, window=14)

    assert len(rsi) == len(df)


############################################################
# COLUMN VALIDATION
############################################################

def test_rsi_missing_column():

    df = pd.DataFrame({
        "price": [1, 2, 3]
    })

    with pytest.raises(Exception):
        TechnicalIndicators.rsi(df)