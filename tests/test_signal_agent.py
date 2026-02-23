import pytest
from core.agent.signal_agent import SignalAgent


def make_base_row():
    return {
        "score": 0.65,
        "rank_pct": 0.80,
        "signal": "LONG",
        "volatility": 0.02,
        "rsi": 55.0,
        "ema_ratio": 1.02,
        "momentum_20_z": 0.5,
        "regime_feature": 0.0,
    }


def make_prob_stats(std=0.10):
    return {
        "mean": 0.55,
        "std": std,
        "min": 0.40,
        "max": 0.75,
    }


# -------------------------------------------------------
# BASIC STRUCTURE TEST
# -------------------------------------------------------

def test_signal_agent_structure():

    agent = SignalAgent()
    row = make_base_row()
    stats = make_prob_stats()

    result = agent.analyze(row=row, probability_stats=stats)

    assert "confidence" in result
    assert "volatility_regime" in result
    assert "trend" in result
    assert "momentum_state" in result
    assert "warnings" in result
    assert "explanation" in result


# -------------------------------------------------------
# HIGH CONFIDENCE TEST
# -------------------------------------------------------

def test_high_confidence():

    agent = SignalAgent()
    row = make_base_row()
    row["score"] = 0.80

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert result["confidence"] == "high"


# -------------------------------------------------------
# LOW DISPERSION WARNING
# -------------------------------------------------------

def test_low_dispersion_warning():

    agent = SignalAgent()
    row = make_base_row()

    result = agent.analyze(
        row=row,
        probability_stats=make_prob_stats(std=0.01)
    )

    assert "Low cross-sectional dispersion detected." in result["warnings"]


# -------------------------------------------------------
# HIGH VOLATILITY REGIME
# -------------------------------------------------------

def test_high_volatility_regime():

    agent = SignalAgent()
    row = make_base_row()
    row["regime_feature"] = 1.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert result["volatility_regime"] == "high_volatility"
    assert "High volatility regime detected." in result["warnings"]


# -------------------------------------------------------
# RSI EXTREME CASES
# -------------------------------------------------------

def test_rsi_overbought():

    agent = SignalAgent()
    row = make_base_row()
    row["rsi"] = 80.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert "RSI indicates overbought condition." in result["warnings"]


def test_rsi_oversold():

    agent = SignalAgent()
    row = make_base_row()
    row["rsi"] = 20.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert "RSI indicates oversold condition." in result["warnings"]


# -------------------------------------------------------
# MOMENTUM CONTRADICTION
# -------------------------------------------------------

def test_momentum_contradicts_long():

    agent = SignalAgent()
    row = make_base_row()
    row["signal"] = "LONG"
    row["momentum_20_z"] = -1.5

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert "Momentum contradicts LONG signal." in result["warnings"]


def test_momentum_contradicts_short():

    agent = SignalAgent()
    row = make_base_row()
    row["signal"] = "SHORT"
    row["momentum_20_z"] = 1.5

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert "Momentum contradicts SHORT signal." in result["warnings"]


# -------------------------------------------------------
# TREND DETECTION
# -------------------------------------------------------

def test_bullish_trend():

    agent = SignalAgent()
    row = make_base_row()
    row["ema_ratio"] = 1.10

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert result["trend"] == "bullish"


def test_bearish_trend():

    agent = SignalAgent()
    row = make_base_row()
    row["ema_ratio"] = 0.90

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert result["trend"] == "bearish"