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


# ---------------------------------------------------
# SAFE ENVIRONMENT (NO PRODUCTION TOUCH)
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def test_environment(monkeypatch):

    monkeypatch.setenv("REDIS_HOST", "invalid-host")
    monkeypatch.setenv("MAX_CONCURRENT_INFERENCES", "2")
    monkeypatch.setenv("MAX_BATCH_SIZE", "3")
    monkeypatch.setenv("MARKETSENTINEL_TEST_MODE", "1")
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("DRIFT_HARD_FAIL", "false")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "30")
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
# SAFE NETWORK BLOCK (ALLOW LOCAL EVENT LOOP)
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def disable_external_network(monkeypatch):
    """
    Block external network access but allow:
    - localhost
    - internal socketpair (asyncio)
    """

    original_socket = socket.socket

    def guarded_socket(*args, **kwargs):

        # Allow internal event loop sockets
        if args and args[0] in (socket.AF_INET, socket.AF_INET6):
            return original_socket(*args, **kwargs)

        # Allow socketpair internally
        if hasattr(socket, "socketpair"):
            try:
                return original_socket(*args, **kwargs)
            except Exception:
                pass

        raise RuntimeError("External network access disabled in test mode")

    monkeypatch.setattr(socket, "socket", guarded_socket)