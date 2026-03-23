import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time

from app.main import app

# ---------------------------------------------------
# MOCK SNAPSHOT matches pipeline v5.6 shape
# /agent/explain reads from _signal_details[ticker]
# ---------------------------------------------------

MOCK_SNAPSHOT = {
    "meta": {
        "model_version": "xgb_test",
        "drift_state": "none",
        "long_signals": 1,
        "short_signals": 0,
        "avg_hybrid_score": 0.5,
        "latency_ms": 30.0,
    },
    "executive_summary": {
        "top_5_tickers": ["NVDA"],
        "portfolio_bias": "LONG_BIASED",
        "gross_exposure": 0.12,
        "net_exposure": 0.12,
    },
    "snapshot": {
        "snapshot_date": "2026-01-01",
        "model_version": "xgb_test",
        "drift": {
            "drift_detected": False,
            "severity_score": 0,
            "drift_state": "none",
            "exposure_scale": 1.0,
            "drift_confidence": 0.0,
        },
        "signals": [
            {
                "ticker": "NVDA",
                "date": "2026-01-01",
                "raw_model_score": 0.82,
                "hybrid_consensus_score": 0.74,
                "weight": 0.12,
            }
        ],
    },
    "_signal_details": {
        "NVDA": {
            "signal_agent": {
                "score": 0.82,
                "confidence": 0.74,
                "signals": {"signal": "LONG"},
                "governance_score": 82,
                "risk_level": "medium",
                "warnings": ["RSI overbought (72.3)"],
                "explanation": "Strong momentum with positive EMA structure.",
            },
            "technical_agent": {
                "score": 0.71,
                "confidence": 0.68,
                "bias": "bullish",
                "signals": {"volatility_regime": "normal", "bias": "bullish"},
                "warnings": [],
            },
        }
    },
    "_political": {"political_risk_score": 0.1, "political_risk_label": "LOW", "top_events": []},
    "_portfolio": {"score": 0.8, "approved_trades": 1, "rejected_trades": 0},
}


@pytest.fixture
def client_with_snapshot():
    """TestClient with mock cache returning MOCK_SNAPSHOT."""
    cache = MagicMock()
    cache.get.side_effect = lambda key: (
        MOCK_SNAPSHOT if key == "ms:background_snapshot:latest" else None
    )
    cache.ping.return_value = True

    model_loader = MagicMock()
    model_loader.is_loaded.return_value = True
    model_loader.version = "xgb_test"
    model_loader.artifact_hash = "a" * 64

    app.state.cache = cache
    app.state.model_loader = model_loader
    app.state.startup_time = time.time() - 60

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def owner_cookies():
    from app.core.auth.jwt_handler import create_owner_token
    token = create_owner_token("test_owner")
    return {"ms_token": token}


# ---------------------------------------------------
# TESTS
# ---------------------------------------------------

class TestAgentExplain:

    def test_explain_returns_200(self, client_with_snapshot, owner_cookies):
        resp = client_with_snapshot.get(
            "/agent/explain?ticker=NVDA",
            cookies=owner_cookies,
        )
        assert resp.status_code == 200

    def test_explain_response_has_data_wrapper(self, client_with_snapshot, owner_cookies):
        """FIX: Response is wrapped in { success, data: {...} } not flat."""
        resp = client_with_snapshot.get(
            "/agent/explain?ticker=NVDA",
            cookies=owner_cookies,
        )
        body = resp.json()
        assert "success" in body
        assert "data" in body
        assert body["success"] is True

    def test_explain_data_has_required_fields(self, client_with_snapshot, owner_cookies):
        """FIX: Fields read from data wrapper not root."""
        resp = client_with_snapshot.get(
            "/agent/explain?ticker=NVDA",
            cookies=owner_cookies,
        )
        data = resp.json()["data"]
        assert data["ticker"] == "NVDA"
        assert "raw_model_score" in data
        assert "hybrid_consensus_score" in data
        assert "signal" in data
        assert "weight" in data

    def test_explain_signal_direction(self, client_with_snapshot, owner_cookies):
        resp = client_with_snapshot.get(
            "/agent/explain?ticker=NVDA",
            cookies=owner_cookies,
        )
        data = resp.json()["data"]
        assert data["signal"] in ("LONG", "SHORT", "NEUTRAL")

    def test_explain_no_agents_sub_object_on_signals(self, client_with_snapshot, owner_cookies):
        """
        Signals in the snapshot no longer have an agents sub-object.
        Agent details come from _signal_details separately.
        """
        resp = client_with_snapshot.get(
            "/agent/explain?ticker=NVDA",
            cookies=owner_cookies,
        )
        data = resp.json()["data"]
        # The response itself has agent-level fields (confidence, governance_score etc.)
        # but the raw signal row does NOT have an agents key
        assert "agents" not in data

    def test_explain_unknown_ticker_returns_404(self, client_with_snapshot, owner_cookies):
        resp = client_with_snapshot.get(
            "/agent/explain?ticker=FAKEXYZ",
            cookies=owner_cookies,
        )
        assert resp.status_code == 404

    def test_explain_missing_ticker_returns_400(self, client_with_snapshot, owner_cookies):
        resp = client_with_snapshot.get(
            "/agent/explain",
            cookies=owner_cookies,
        )
        assert resp.status_code == 400

    def test_explain_post_also_works(self, client_with_snapshot, owner_cookies):
        """Route supports both GET and POST."""
        resp = client_with_snapshot.post(
            "/agent/explain",
            json={"ticker": "NVDA"},
            cookies=owner_cookies,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["ticker"] == "NVDA"

    def test_explain_no_auth_still_processes(self, client_with_snapshot):
        """No auth = unauthenticated, but route still responds (middleware passes through)."""
        resp = client_with_snapshot.get("/agent/explain?ticker=NVDA")
        # May return 200 or 401 depending on middleware config — just not 500
        assert resp.status_code != 500

    def test_explain_warnings_list(self, client_with_snapshot, owner_cookies):
        resp = client_with_snapshot.get(
            "/agent/explain?ticker=NVDA",
            cookies=owner_cookies,
        )
        data = resp.json()["data"]
        assert isinstance(data.get("warnings", []), list)

    def test_explain_llm_null_when_disabled(self, client_with_snapshot, owner_cookies):
        """LLM section is None when LLM_ENABLED=false (default)."""
        resp = client_with_snapshot.get(
            "/agent/explain?ticker=NVDA",
            cookies=owner_cookies,
        )
        data = resp.json()["data"]
        # llm field should be None or have llm_enabled=False
        llm = data.get("llm")
        if llm is not None:
            assert llm.get("llm_enabled") is False


class TestAgentAgents:

    def test_agents_list_returns_200(self, client_with_snapshot):
        resp = client_with_snapshot.get("/agent/agents")
        assert resp.status_code == 200

    def test_agents_list_has_four_agents(self, client_with_snapshot):
        resp = client_with_snapshot.get("/agent/agents")
        agents = resp.json()["agents"]
        assert len(agents) == 4
        assert "signal_agent" in agents
        assert "technical_risk_agent" in agents
        assert "portfolio_decision_agent" in agents
        assert "political_risk_agent" in agents

    def test_agents_list_no_inference(self, client_with_snapshot):
        """FIX: /agent/agents is now static — does not trigger model inference."""
        # If it ran inference on a mock it would fail with model errors
        # The fact it returns 200 confirms it's static
        resp = client_with_snapshot.get("/agent/agents")
        assert resp.status_code == 200