import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app


client = TestClient(app)


# -------------------------------------------------------
# MOCK SNAPSHOT
# -------------------------------------------------------

def mock_snapshot():

    return {
        "probability_stats": {
            "mean": 0.60,
            "std": 0.10,
        },
        "signals": [
            {
                "date": "2024-01-01",
                "ticker": "AAPL",
                "score": 0.72,
                "signal": "LONG",
                "agent": {
                    "strength_score": 82,
                    "risk_level": "low",
                    "confidence": "high",
                    "volatility_regime": "normal",
                    "trend": "bullish",
                    "momentum_state": "strong_positive",
                    "macro_regime": "neutral",
                    "warnings": [],
                    "reasoning": [],
                    "explanation": "Mock explanation"
                }
            }
        ]
    }


# -------------------------------------------------------
# SUCCESS CASE
# -------------------------------------------------------

@patch("app.api.routes.agent.get_explainer")
@patch("app.api.routes.agent.InferencePipeline")
def test_agent_explanation_success(mock_pipeline_cls, mock_explainer):

    # Mock pipeline
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.run_snapshot.return_value = mock_snapshot()

    # Mock LLM explanation
    mock_explainer.return_value.explain.return_value = "LLM explanation"

    response = client.post("/agent/explain", params={"ticker": "AAPL"})

    assert response.status_code == 200

    data = response.json()

    assert data["ticker"] == "AAPL"
    assert data["signal"] == "LONG"
    assert "agent" in data
    assert "llm" in data

    agent_data = data["agent"]

    assert agent_data["strength_score"] == 82
    assert agent_data["confidence"] == "high"
    assert agent_data["trend"] == "bullish"
    assert agent_data["momentum_state"] == "strong_positive"


# -------------------------------------------------------
# INVALID TICKER
# -------------------------------------------------------

def test_invalid_ticker():

    response = client.post("/agent/explain", params={"ticker": ""})

    assert response.status_code == 400


# -------------------------------------------------------
# NO SIGNAL GENERATED
# -------------------------------------------------------

@patch("app.api.routes.agent.InferencePipeline")
def test_no_signal(mock_pipeline_cls):

    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.run_snapshot.return_value = {
        "probability_stats": {},
        "signals": []
    }

    response = client.post("/agent/explain", params={"ticker": "AAPL"})

    assert response.status_code == 404