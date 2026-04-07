"""
Tests for AuthMiddleware — owner-only routes, demo quota,
unauthenticated requests.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from starlette.testclient import TestClient
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.auth.middleware import AuthMiddleware, _is_owner_only, _get_feature_group


# =====================================================
# OWNER-ONLY PATH DETECTION
# =====================================================

class TestOwnerOnlyPaths:

    def test_admin_path_is_owner_only(self):
        assert _is_owner_only("/admin/sync") is True
        assert _is_owner_only("/admin") is True
        assert _is_owner_only("/admin/retrain") is True

    def test_ic_stats_is_owner_only(self):
        assert _is_owner_only("/model/ic-stats") is True
        assert _is_owner_only("/model/ic-stats?days=30") is True

    def test_diagnostics_is_owner_only(self):
        assert _is_owner_only("/model/diagnostics") is True

    def test_public_paths_not_owner_only(self):
        assert _is_owner_only("/health/live") is False
        assert _is_owner_only("/drift") is False
        assert _is_owner_only("/portfolio") is False
        assert _is_owner_only("/model/info") is False
        assert _is_owner_only("/model/feature-importance") is False


# =====================================================
# FEATURE GROUP MAPPING
# =====================================================

class TestFeatureGroupMapping:

    def test_snapshot_paths(self):
        assert _get_feature_group("/predict/live-snapshot") == "snapshot"
        assert _get_feature_group("/snapshot") == "snapshot"

    def test_portfolio_path(self):
        assert _get_feature_group("/portfolio") == "portfolio"

    def test_drift_path(self):
        assert _get_feature_group("/drift") == "drift"

    def test_performance_path(self):
        assert _get_feature_group("/performance") == "performance"

    def test_agent_paths(self):
        assert _get_feature_group("/agent/explain") == "agent"
        assert _get_feature_group("/agent/political-risk") == "agent"

    def test_signals_paths(self):
        assert _get_feature_group("/equity") == "signals"
        assert _get_feature_group("/model/feature-importance") == "signals"

    def test_free_paths_return_none(self):
        assert _get_feature_group("/health/live") is None
        assert _get_feature_group("/health/ready") is None
        assert _get_feature_group("/universe") is None
        assert _get_feature_group("/auth/me") is None
        assert _get_feature_group("/model/info") is None
        assert _get_feature_group("/docs") is None
