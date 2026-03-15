import pytest
from copy import deepcopy

from core.agent.signal_agent import SignalAgent


############################################################
# FIXED TEST INPUTS
############################################################

def make_base_context():

    return {
        "row": {
            "raw_model_score": 0.65,
            "signal": "LONG",
            "volatility": 0.02,
            "rsi": 55.0,
            "ema_ratio": 1.02,
            "momentum_20_z": 0.5,
            "regime_feature": 0.0,
        },
        "probability_stats": {
            "mean": 0.55,
            "std": 0.10
        },
        "drift_state": None,
        "political_risk_label": None
    }


############################################################
# STRUCTURE CONTRACT
############################################################

def test_signal_agent_structure():

    agent = SignalAgent()

    result = agent.analyze(make_base_context())

    required_keys = {
        "signal",
        "confidence_numeric",
        "technical_score",
        "agent_score",
        "risk_level",
        "volatility_regime",
        "alignment_score",
        "position_size_hint",
        "trade_approved",
        "governance_score",
        "reasoning",
        "warnings",
        "explanation",
        "hybrid"
    }

    assert required_keys.issubset(result.keys())

    assert isinstance(result["warnings"], list)
    assert isinstance(result["reasoning"], list)
    assert isinstance(result["explanation"], str)


############################################################
# DETERMINISM
############################################################

def test_agent_is_deterministic():

    agent = SignalAgent()

    context = make_base_context()

    r1 = agent.analyze(context)
    r2 = agent.analyze(context)

    assert r1 == r2


############################################################
# CONFIDENCE RANGE
############################################################

def test_confidence_range():

    agent = SignalAgent()

    result = agent.analyze(make_base_context())

    assert 0.0 <= result["confidence_numeric"] <= 1.0


############################################################
# CONFIDENCE MONOTONICITY
############################################################

def test_confidence_monotonicity():

    agent = SignalAgent()

    low = deepcopy(make_base_context())
    low["row"]["raw_model_score"] = 0.2

    high = deepcopy(make_base_context())
    high["row"]["raw_model_score"] = 2.0

    r_low = agent.analyze(low)
    r_high = agent.analyze(high)

    assert r_high["confidence_numeric"] > r_low["confidence_numeric"]


############################################################
# LOW DISPERSION WARNING
############################################################

def test_low_dispersion_warning():

    agent = SignalAgent()

    context = make_base_context()
    context["probability_stats"]["std"] = 0.01

    result = agent.analyze(context)

    assert any("dispersion" in w.lower() for w in result["warnings"])


############################################################
# HIGH VOLATILITY REGIME
############################################################

def test_high_volatility_regime():

    agent = SignalAgent()

    context = make_base_context()
    context["row"]["regime_feature"] = 2.0

    result = agent.analyze(context)

    assert result["volatility_regime"] == "high_volatility"


############################################################
# RSI EXTREMES
############################################################

def test_rsi_overbought():

    agent = SignalAgent()

    context = make_base_context()
    context["row"]["rsi"] = 80

    result = agent.analyze(context)

    assert any("overbought" in w.lower() for w in result["warnings"])


def test_rsi_oversold():

    agent = SignalAgent()

    context = make_base_context()
    context["row"]["rsi"] = 20
    context["row"]["signal"] = "SHORT"

    result = agent.analyze(context)

    assert any("oversold" in w.lower() for w in result["warnings"])


############################################################
# MOMENTUM CONTRADICTION
############################################################

def test_momentum_contradiction_long():

    agent = SignalAgent()

    context = make_base_context()
    context["row"]["signal"] = "LONG"
    context["row"]["momentum_20_z"] = -2.0

    result = agent.analyze(context)

    assert any("contradict" in w.lower() for w in result["warnings"])


def test_momentum_contradiction_short():

    agent = SignalAgent()

    context = make_base_context()
    context["row"]["signal"] = "SHORT"
    context["row"]["momentum_20_z"] = 2.0

    result = agent.analyze(context)

    assert any("contradict" in w.lower() for w in result["warnings"])


############################################################
# POSITION SIZE RANGE
############################################################

def test_position_size_range():

    agent = SignalAgent()

    result = agent.analyze(make_base_context())

    assert 0.0 <= result["position_size_hint"] <= 1.0


############################################################
# POLITICAL RISK OVERRIDE
############################################################

def test_political_risk_override():

    agent = SignalAgent()

    context = make_base_context()
    context["political_risk_label"] = "CRITICAL"

    result = agent.analyze(context)

    assert result["signal"] == "NEUTRAL"
    assert result["trade_approved"] is False