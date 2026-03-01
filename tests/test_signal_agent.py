import pytest
from core.agent.signal_agent import SignalAgent


# ======================================================
# FIXED TEST INPUTS
# ======================================================

def make_base_row():
    return {
        "score": 0.65,
        "signal": "LONG",
        "volatility": 0.02,
        "rsi": 55.0,
        "ema_ratio": 1.02,
        "momentum_20_z": 0.5,
        "regime_feature": 0.0,
        "breadth": 0.5,
    }


def make_prob_stats(std=0.10):
    return {
        "mean": 0.55,
        "std": std,
    }


# ======================================================
# STRUCTURE CONTRACT LOCK
# ======================================================

def test_signal_agent_structure():

    agent = SignalAgent()
    result = agent.analyze(
        row=make_base_row(),
        probability_stats=make_prob_stats()
    )

    required_keys = {
        "signal",
        "confidence",
        "confidence_numeric",
        "strength_score",
        "risk_level",
        "volatility_regime",
        "trend",
        "momentum_state",
        "macro_regime",
        "reasoning",
        "warnings",
        "explanation",
    }

    assert required_keys.issubset(result.keys())


# ======================================================
# DETERMINISM
# ======================================================

def test_agent_is_deterministic():

    agent = SignalAgent()

    row = make_base_row()
    stats = make_prob_stats()

    r1 = agent.analyze(row=row, probability_stats=stats)
    r2 = agent.analyze(row=row, probability_stats=stats)

    assert r1 == r2


# ======================================================
# CONFIDENCE TIERS
# ======================================================

def test_confidence_high():

    agent = SignalAgent()
    row = make_base_row()
    row["score"] = 1.5

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert result["confidence"] in {"high", "very_high"}
    assert 0.0 <= result["confidence_numeric"] <= 1.0


def test_confidence_monotonicity():

    agent = SignalAgent()

    row_low = make_base_row()
    row_low["score"] = 0.3

    row_high = make_base_row()
    row_high["score"] = 1.5

    r_low = agent.analyze(row=row_low, probability_stats=make_prob_stats())
    r_high = agent.analyze(row=row_high, probability_stats=make_prob_stats())

    assert r_high["confidence_numeric"] > r_low["confidence_numeric"]


# ======================================================
# LOW DISPERSION WARNING
# ======================================================

def test_low_dispersion_warning():

    agent = SignalAgent()

    result = agent.analyze(
        row=make_base_row(),
        probability_stats=make_prob_stats(std=0.01)
    )

    assert any("dispersion" in w.lower() for w in result["warnings"])


# ======================================================
# HIGH VOLATILITY REGIME
# ======================================================

def test_high_volatility_regime():

    agent = SignalAgent()
    row = make_base_row()
    row["regime_feature"] = 2.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert result["volatility_regime"] == "high_volatility"
    assert any("volatility" in w.lower() for w in result["warnings"])


# ======================================================
# RSI EXTREMES
# ======================================================

def test_rsi_overbought():

    agent = SignalAgent()
    row = make_base_row()
    row["rsi"] = 80.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert any("overbought" in w.lower() for w in result["warnings"])


def test_rsi_oversold():

    agent = SignalAgent()
    row = make_base_row()
    row["rsi"] = 20.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert any("oversold" in r.lower() for r in result["reasoning"])


# ======================================================
# MOMENTUM CONTRADICTION
# ======================================================

def test_momentum_contradiction_long():

    agent = SignalAgent()
    row = make_base_row()
    row["signal"] = "LONG"
    row["momentum_20_z"] = -2.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert any("contradict" in w.lower() for w in result["warnings"])


def test_momentum_contradiction_short():

    agent = SignalAgent()
    row = make_base_row()
    row["signal"] = "SHORT"
    row["momentum_20_z"] = 2.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert any("contradict" in w.lower() for w in result["warnings"])


# ======================================================
# TREND DETECTION
# ======================================================

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


# ======================================================
# STRENGTH SCORE
# ======================================================

def test_strength_score_range():

    agent = SignalAgent()
    result = agent.analyze(
        row=make_base_row(),
        probability_stats=make_prob_stats()
    )

    assert 0 <= result["strength_score"] <= 100


def test_strength_score_monotonicity():

    agent = SignalAgent()

    row_low = make_base_row()
    row_low["score"] = 0.3

    row_high = make_base_row()
    row_high["score"] = 1.2

    r_low = agent.analyze(row=row_low, probability_stats=make_prob_stats())
    r_high = agent.analyze(row=row_high, probability_stats=make_prob_stats())

    assert r_high["strength_score"] > r_low["strength_score"]


# ======================================================
# RISK CLASSIFICATION
# ======================================================

def test_risk_low_for_strong_signal():

    agent = SignalAgent()
    row = make_base_row()
    row["score"] = 2.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert result["risk_level"] in {"low", "moderate"}


def test_risk_elevated_under_volatility():

    agent = SignalAgent()
    row = make_base_row()
    row["regime_feature"] = 2.0

    result = agent.analyze(row=row, probability_stats=make_prob_stats())

    assert result["risk_level"] in {"elevated", "high", "moderate"}