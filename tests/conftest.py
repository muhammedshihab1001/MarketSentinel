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

    try:
        import torch
        torch.manual_seed(42)
    except Exception:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except Exception:
        pass


# ---------------------------------------------------
# SAFE ENVIRONMENT
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def test_environment(monkeypatch):
    """
    Prevent tests from touching production services.
    """

    monkeypatch.setenv("REDIS_HOST", "invalid-host")
    monkeypatch.setenv("MAX_CONCURRENT_INFERENCES", "2")
    monkeypatch.setenv("MAX_BATCH_SIZE", "3")


# ---------------------------------------------------
# ARTIFACT ISOLATION
# ---------------------------------------------------

@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    """
    Redirect ALL artifact paths into pytest temp dir.
    """

    artifacts_dir = tmp_path / "artifacts"
    data_dir = tmp_path / "data"

    artifacts_dir.mkdir()
    data_dir.mkdir()

    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    return tmp_path


# ---------------------------------------------------
# FAST TEST MODE FLAG
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def fast_test_mode(monkeypatch):
    """
    Lets your code detect it is running under pytest.

    Useful for:
    - disabling heavy model loads
    - skipping walk-forward
    """

    monkeypatch.setenv("MARKETSENTINEL_TEST_MODE", "1")


# ---------------------------------------------------
# LOGGING CONTROL
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def silence_logs():
    import logging

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
