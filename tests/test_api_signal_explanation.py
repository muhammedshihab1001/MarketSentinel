import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app


client = TestClient(app)


# -------------------------------------------------------
# MOCK SNAPSHOT OUTPUT
# -------------------------------------------------------

def mock_snapshot_output():

    return {
        "snapshot_date": "2024-01-01",
        "universe_size": 1,
        "probability_stats": {
            "mean": 0.60,
            "std": 0.10,
            "min": 0.40,
            "max": 0.80,
        },
        "signals": [
            {
                "date": "2024-01-01",
                "ticker": "AAPL",
                "score": 0.72,
                "rank_pct": 0.85,
                "signal": "LONG",
                "weight": 0.15,
                "agent": {
                    "strength_score": 82.5,
                    "risk_level": "normal",
                    "confidence": "high",
                    "volatility_regime": "normal",
                    "trend": "bullish",
                    "momentum_state": "strong_positive",
                    "warnings": [],
                    "explanation": "Mock explanation"
                }
            }
        ]
    }


# -------------------------------------------------------
# SUCCESS CASE
# -------------------------------------------------------

@patch("app.api.routes.predict.get_pipeline")
@patch("app.api.routes.predict.get_loader")
def test_signal_explanation_success(mock_loader, mock_pipeline):

    # Mock loader attributes
    mock_loader.return_value.xgb_version = "test_model"
    mock_loader.return_value.schema_signature = "abc123"
    mock_loader.return_value.dataset_hash = "dataset_hash"
    mock_loader.return_value.training_code_hash = "train_hash"
    mock_loader.return_value.artifact_hash = "artifact_hash"

    # Mock pipeline snapshot
    mock_pipeline.return_value.run_snapshot.return_value = mock_snapshot_output()

    response = client.get("/signal-explanation/AAPL")

    assert response.status_code == 200

    data = response.json()

    assert "meta" in data
    assert "explanation" in data

    explanation = data["explanation"]

    assert explanation["ticker"] == "AAPL"
    assert explanation["signal"] == "LONG"
    assert explanation["strength_score"] == 82.5
    assert explanation["risk_level"] == "normal"
    assert explanation["confidence"] == "high"
    assert explanation["trend"] == "bullish"
    assert explanation["momentum_state"] == "strong_positive"


# -------------------------------------------------------
# INVALID TICKER
# -------------------------------------------------------

def test_invalid_ticker():

    response = client.get("/signal-explanation/INVALID$$$")

    assert response.status_code == 400


# -------------------------------------------------------
# NO SIGNAL GENERATED
# -------------------------------------------------------

@patch("app.api.routes.predict.get_pipeline")
@patch("app.api.routes.predict.get_loader")
def test_no_signal(mock_loader, mock_pipeline):

    mock_loader.return_value.xgb_version = "test_model"
    mock_loader.return_value.schema_signature = "abc123"
    mock_loader.return_value.dataset_hash = "dataset_hash"
    mock_loader.return_value.training_code_hash = "train_hash"
    mock_loader.return_value.artifact_hash = "artifact_hash"

    mock_pipeline.return_value.run_snapshot.return_value = {
        "signals": []
    }

    response = client.get("/signal-explanation/AAPL")

    assert response.status_code == 404