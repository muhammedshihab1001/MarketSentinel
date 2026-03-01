import sys
import os
import pytest
import numpy as np
import random


# ---------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------
# GLOBAL DETERMINISM (CRITICAL FOR ML TESTS)
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def set_test_seeds():
    np.random.seed(42)
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"


# ---------------------------------------------------
# SAFE ENVIRONMENT (NO PRODUCTION TOUCH)
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def test_environment(monkeypatch):

    # Prevent real Redis
    monkeypatch.setenv("REDIS_HOST", "invalid-host")

    # Force small resource caps
    monkeypatch.setenv("MAX_CONCURRENT_INFERENCES", "2")
    monkeypatch.setenv("MAX_BATCH_SIZE", "3")

    # Enable test mode flag
    monkeypatch.setenv("MARKETSENTINEL_TEST_MODE", "1")

    # Disable LLM in tests
    monkeypatch.setenv("LLM_ENABLED", "false")

    # Disable drift hard fail unless explicitly tested
    monkeypatch.setenv("DRIFT_HARD_FAIL", "false")

    # Short cache TTL for safety
    monkeypatch.setenv("CACHE_TTL_SECONDS", "30")

    # Force deterministic XGBoost threading
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


# ---------------------------------------------------
# ARTIFACT ISOLATION
# ---------------------------------------------------

@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):

    artifacts_dir = tmp_path / "artifacts"
    data_dir = tmp_path / "data"

    artifacts_dir.mkdir()
    data_dir.mkdir()

    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    return tmp_path


# ---------------------------------------------------
# OPTIONAL: BLOCK NETWORK ACCESS
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def disable_network(monkeypatch):
    """
    Prevent accidental external API calls during tests.
    """

    def guard(*args, **kwargs):
        raise RuntimeError("Network access disabled in test mode")

    monkeypatch.setattr("socket.socket", guard, raising=False)