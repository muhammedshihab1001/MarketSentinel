import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import socket

from app.main import app


############################################################
# MOCK SNAPSHOT (matches actual pipeline.run_snapshot output)
############################################################

def mock_snapshot():

    return {
        "snapshot_date": "2024-01-01",
        "model_version": "test_v1",
        "gross_exposure": 0.5,
        "net_exposure": 0.1,
        "drift": {
            "drift_state": "none",
            "severity_score": 0,
            "exposure_scale": 1.0,
        },
        "signals": [
            {
                "date": "2024-01-01",
                "ticker": "AAPL",
                "raw_model_score": 0.72,
                "agent_score": 0.65,
                "technical_score": 0.55,
                "hybrid_consensus_score": 0.60,
                "weight": 0.10,
                "agents": {
                    "signal_agent": {
                        "signal": "LONG",
                        "alpha_strength": 0.72,
                        "confidence_numeric": 0.80,
                        "governance_score": 82,
                        "risk_level": "low",
                        "volatility_regime": "normal",
                        "trade_approved": True,
                        "drift_flag": False,
                        "warnings": [],
                        "reasoning": ["Test reasoning"],
                        "explanation": "Mock explanation",
                        "hybrid": {
                            "score": 0.65,
                            "confidence": 0.80,
                        },
                    },
                    "technical_agent": {
                        "score": 0.55,
                        "confidence": 0.60,
                        "bias": "bullish",
                        "warnings": [],
                        "component_scores": {
                            "momentum": 0.7,
                            "ema": 0.5,
                            "rsi": 1.0,
                        },
                    },
                },
            }
        ],
        "top_5": [],
        "decision_report": {},
    }


############################################################
# SUCCESS CASE
############################################################

@patch("app.api.routes.agent.get_explainer")
@patch("app.api.routes.predict._pipeline")
def test_agent_explanation_success(mock_pipeline, mock_explainer, monkeypatch):

    # Restore original socket for TestClient compatibility
    monkeypatch.setattr(socket, "socket", socket.socket)

    # Set up pipeline mock
    pipeline_mock = MagicMock()
    pipeline_mock.run_snapshot.return_value = mock_snapshot()

    # Patch the get_pipeline function to return our mock
    with patch("app.api.routes.predict.get_pipeline", return_value=pipeline_mock):
        with patch("app.api.routes.agent.get_pipeline", return_value=pipeline_mock):

            mock_explainer.return_value.explain.return_value = {
                "llm_enabled": False,
                "message": "LLM disabled",
            }

            client = TestClient(app)

            response = client.post("/agent/explain", params={"ticker": "AAPL"})

    assert response.status_code == 200

    payload = response.json()

    data = payload.get("data", payload)

    assert data["ticker"] == "AAPL"
    assert data["signal"] == "LONG"
    assert data["confidence_numeric"] == 0.80
    assert data["risk_level"] == "low"
    assert data["volatility_regime"] == "normal"
    assert data["hybrid_consensus_score"] == 0.60
    assert data["raw_model_score"] == 0.72


############################################################
# INVALID TICKER
############################################################

def test_invalid_ticker(monkeypatch):

    monkeypatch.setattr(socket, "socket", socket.socket)

    client = TestClient(app)

    response = client.post("/agent/explain", params={"ticker": ""})

    assert response.status_code == 400


############################################################
# NO SIGNAL GENERATED
############################################################

def test_no_signal(monkeypatch):

    monkeypatch.setattr(socket, "socket", socket.socket)

    empty_snapshot = {
        "snapshot_date": "2024-01-01",
        "model_version": "test_v1",
        "gross_exposure": 0.0,
        "net_exposure": 0.0,
        "drift": {},
        "signals": [],
        "top_5": [],
        "decision_report": {},
    }

    pipeline_mock = MagicMock()
    pipeline_mock.run_snapshot.return_value = empty_snapshot

    with patch("app.api.routes.agent.get_pipeline", return_value=pipeline_mock):

        client = TestClient(app)

        response = client.post("/agent/explain", params={"ticker": "AAPL"})

    assert response.status_code == 404