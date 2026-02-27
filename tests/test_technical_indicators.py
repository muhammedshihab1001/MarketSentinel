import pandas as pd
import numpy as np

from core.indicators.technical_indicators import TechnicalIndicators


# ---------------------------------------------------
# BASIC EXECUTION
# ---------------------------------------------------

def test_rsi_runs_without_crashing():

    df = pd.DataFrame({
        "Close": np.linspace(100, 120, 50)
    })

    rsi = TechnicalIndicators.rsi(df)

    assert len(rsi) == len(df)


# ---------------------------------------------------
# BOUNDS CHECK (VERY IMPORTANT)
# RSI MUST ALWAYS BE BETWEEN 0–100
# ---------------------------------------------------

def test_rsi_bounds():

    df = pd.DataFrame({
        "Close": np.random.uniform(90, 110, 100)
    })

    rsi = TechnicalIndicators.rsi(df).dropna()

    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


# ---------------------------------------------------
# MONOTONIC UP MARKET
# RSI SHOULD BE HIGH
# ---------------------------------------------------

def test_rsi_uptrend():

    df = pd.DataFrame({
        "Close": np.arange(100, 200)
    })

    rsi = TechnicalIndicators.rsi(df)

    assert rsi.iloc[-1] > 60


# ---------------------------------------------------
# MONOTONIC DOWN MARKET
# RSI SHOULD BE LOW
# ---------------------------------------------------

def test_rsi_downtrend():

    df = pd.DataFrame({
        "Close": np.arange(200, 100, -1)
    })

    rsi = TechnicalIndicators.rsi(df)

    assert rsi.iloc[-1] < 40


# ---------------------------------------------------
# FLAT MARKET
# SHOULD NOT CRASH OR DIVIDE BY ZERO
# ---------------------------------------------------

def test_rsi_flat_market():

    df = pd.DataFrame({
        "Close": [100] * 100
    })

    rsi = TechnicalIndicators.rsi(df)

    assert not rsi.isnull().all()


# ---------------------------------------------------
# COLUMN VALIDATION
# ---------------------------------------------------

def test_rsi_missing_column():

    df = pd.DataFrame({
        "price": [1, 2, 3]
    })

    try:
        TechnicalIndicators.rsi(df)
        assert False, "Expected failure for missing column"
    except Exception:
        assert True