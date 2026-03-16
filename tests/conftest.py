import sys
import os
import pytest
import numpy as np
import random
import socket


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

    # Prevent hidden nondeterminism
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ---------------------------------------------------
# SAFE ENVIRONMENT (NO PRODUCTION TOUCH)
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def test_environment(monkeypatch):

    # Disable Redis
    monkeypatch.setenv("REDIS_HOST", "invalid-host")
    monkeypatch.setenv("REDIS_PORT", "6379")

    # Disable LLM
    monkeypatch.setenv("LLM_ENABLED", "false")

    # Disable strict drift failure
    monkeypatch.setenv("DRIFT_HARD_FAIL", "false")

    # Prevent heavy inference
    monkeypatch.setenv("MAX_CONCURRENT_INFERENCES", "2")
    monkeypatch.setenv("MAX_BATCH_SIZE", "3")

    # Reduce cache TTL
    monkeypatch.setenv("CACHE_TTL_SECONDS", "30")

    # Prevent GPU usage in CI
    monkeypatch.setenv("XGB_USE_GPU", "false")

    # Prevent model governance strict failures
    monkeypatch.setenv("MODEL_STRICT_GOVERNANCE", "0")

    # Ensure fallback allowed
    monkeypatch.setenv("MODEL_ALLOW_POINTER_FALLBACK", "1")

    # Mark test mode globally
    monkeypatch.setenv("MARKETSENTINEL_TEST_MODE", "1")


# ---------------------------------------------------
# ARTIFACT ISOLATION
# ---------------------------------------------------

@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):

    artifacts_dir = tmp_path / "artifacts"
    data_dir = tmp_path / "data"

    artifacts_dir.mkdir()
    data_dir.mkdir()

    # Redirect project artifact paths
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    # Prevent model loader from touching real registry
    monkeypatch.setenv("XGB_REGISTRY_DIR", str(artifacts_dir / "xgboost"))

    return tmp_path


# ---------------------------------------------------
# SAFE NETWORK BLOCK
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def disable_external_network(monkeypatch):
    """
    Block external network access while allowing:
    - localhost
    - internal FastAPI TestClient sockets
    """

    original_socket = socket.socket

    class GuardedSocket(socket.socket):

        def connect(self, address):

            host = address[0]

            # allow localhost
            if host in ("127.0.0.1", "localhost"):
                return super().connect(address)

            raise RuntimeError(
                "External network access disabled in test mode"
            )

    monkeypatch.setattr(socket, "socket", GuardedSocket)

    yield

    # restore original socket after test
    monkeypatch.setattr(socket, "socket", original_socket)


# ---------------------------------------------------
# PYTEST GLOBAL CONFIG
# ---------------------------------------------------

def pytest_configure(config):
    """
    Global pytest configuration for MarketSentinel tests.
    Ensures test mode is always enabled.
    """

    os.environ["MARKETSENTINEL_TEST_MODE"] = "1"