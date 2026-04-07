import sys
import os
import pytest
import numpy as np
import random
import socket
from unittest.mock import MagicMock, patch

# ---------------------------------------------------
# FIX: Set JWT_SECRET before importing app.main.
# jwt_handler.py raises RuntimeError at module level
# if JWT_SECRET is missing — kills test collection.
# This must be the FIRST thing in conftest.py.
# ---------------------------------------------------
os.environ.setdefault("JWT_SECRET", "test_secret_for_ci_only_not_production_" + "x" * 32)
os.environ.setdefault("OWNER_PASSWORD_HASH", "$2b$12$placeholder_hash_for_testing_only_xxxxxx")
os.environ.setdefault("LLM_ENABLED", "0")
os.environ.setdefault("REDIS_HOST", "invalid-host-for-tests")
os.environ.setdefault("MODEL_ALLOW_POINTER_FALLBACK", "1")
os.environ.setdefault("DRIFT_HARD_FAIL", "false")
os.environ.setdefault("MODEL_STRICT_GOVERNANCE", "0")
os.environ.setdefault("SKIP_DATA_SYNC", "1")

# ---------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------
# DETERMINISM
# ---------------------------------------------------
np.random.seed(42)
random.seed(42)


# ---------------------------------------------------
# MOCK SNAPSHOT
# Matches pipeline v5.6 output shape.
# meta + executive_summary + snapshot.signals[]
# _signal_details for /agent/explain
# ---------------------------------------------------

MOCK_SNAPSHOT = {
    "meta": {
        "model_version": "xgb_test_20260101_000000",
        "drift_state": "none",
        "long_signals": 3,
        "short_signals": 2,
        "avg_hybrid_score": 0.125,
        "latency_ms": 45.2,
    },
    "executive_summary": {
        "top_5_tickers": ["NVDA", "MSFT", "AAPL", "GOOGL", "JPM"],
        "portfolio_bias": "LONG_BIASED",
        "gross_exposure": 0.42,
        "net_exposure": 0.18,
    },
    "snapshot": {
        "snapshot_date": "2026-01-01",
        "model_version": "xgb_test_20260101_000000",
        "drift": {
            "drift_detected": False,
            "severity_score": 1,
            "drift_state": "none",
            "exposure_scale": 1.0,
            "drift_confidence": 0.05,
        },
        "signals": [
            {
                "ticker": "NVDA",
                "date": "2026-01-01",
                "raw_model_score": 0.82,
                "hybrid_consensus_score": 0.74,
                "weight": 0.12,
            },
            {
                "ticker": "MSFT",
                "date": "2026-01-01",
                "raw_model_score": 0.71,
                "hybrid_consensus_score": 0.65,
                "weight": 0.09,
            },
            {
                "ticker": "AAPL",
                "date": "2026-01-01",
                "raw_model_score": 0.68,
                "hybrid_consensus_score": 0.61,
                "weight": 0.08,
            },
            {
                "ticker": "GOOGL",
                "date": "2026-01-01",
                "raw_model_score": 0.55,
                "hybrid_consensus_score": 0.50,
                "weight": 0.07,
            },
            {
                "ticker": "JPM",
                "date": "2026-01-01",
                "raw_model_score": -0.45,
                "hybrid_consensus_score": -0.40,
                "weight": -0.06,
            },
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
                "warnings": [],
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
    "_political": {
        "political_risk_score": 0.12,
        "political_risk_label": "LOW",
        "top_events": [],
    },
    "_portfolio": {
        "score": 0.78,
        "approved_trades": 5,
        "rejected_trades": 1,
    },
}


# ---------------------------------------------------
# FIXTURES
# ---------------------------------------------------

@pytest.fixture
def mock_snapshot():
    import copy
    return copy.deepcopy(MOCK_SNAPSHOT)


@pytest.fixture
def mock_cache(mock_snapshot):
    cache = MagicMock()
    cache.get.side_effect = lambda key: (
        mock_snapshot if key == "ms:background_snapshot:latest" else None
    )
    cache.set.return_value = True
    cache.ping.return_value = True
    cache._redis = MagicMock()
    cache.enabled = False  # treat as memory-only in tests
    return cache


@pytest.fixture
def mock_model_loader():
    loader = MagicMock()
    loader.is_loaded.return_value = True
    loader.version = "xgb_test_20260101_000000"
    loader.artifact_hash = "a" * 64
    loader.schema_signature = "b" * 64
    loader.feature_names = ["f1", "f2", "f3"]
    loader.metadata = {
        "dataset_hash": "d" * 64,
        "training_code_hash": "e" * 64,
        "feature_checksum": "f" * 64,
    }
    loader.predict.return_value = np.array([0.5, 0.3, -0.2])
    return loader


@pytest.fixture
def test_app(mock_cache, mock_model_loader):
    import time
    from fastapi.testclient import TestClient
    from app.main import app

    app.state.cache = mock_cache
    app.state.model_loader = mock_model_loader
    app.state.startup_time = time.time() - 120

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def owner_token():
    from app.core.auth.jwt_handler import create_owner_token
    return create_owner_token("test_owner")


@pytest.fixture
def demo_token():
    from app.core.auth.jwt_handler import create_demo_token
    return create_demo_token("test_fingerprint_abc123")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring live services"
    )
