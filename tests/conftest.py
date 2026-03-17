import sys
import os
import pytest
import numpy as np
import random
import socket
from unittest.mock import MagicMock, patch


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

    # Skip Yahoo data sync — no live calls in tests
    monkeypatch.setenv("SKIP_DATA_SYNC", "1")

    # Disable prediction writes to DB in tests
    monkeypatch.setenv("STORE_PREDICTIONS", "0")

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

    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("XGB_REGISTRY_DIR", str(artifacts_dir / "xgboost"))

    return tmp_path


# ---------------------------------------------------
# DATABASE MOCK FIXTURE
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """
    Mocks the database layer for all tests by default.

    Tests that actually need a real DB (e.g. integration tests
    running against the CI postgres service) can opt out by
    overriding this fixture locally:

        @pytest.fixture(autouse=False)
        def mock_db():
            pass  # real DB used

    What this mocks:
    - core.db.engine.init_db       — no-op, doesn't try to connect
    - core.db.engine.get_session   — returns a MagicMock session
    - core.db.engine.dispose_engine — no-op
    - core.db.repository.OHLCVRepository — returns empty DataFrames
    - core.db.repository.PredictionRepository — silent no-op writes

    MarketDataService is also patched to return a minimal synthetic
    DataFrame so tests that indirectly trigger data fetches don't
    hang waiting for Yahoo Finance or a real DB.
    """

    # ── DB engine ────────────────────────────────────────
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr(
        "core.db.engine.init_db",
        lambda: None,
    )
    monkeypatch.setattr(
        "core.db.engine.dispose_engine",
        lambda: None,
    )
    monkeypatch.setattr(
        "core.db.engine.get_session",
        lambda: mock_session,
    )

    # ── OHLCV repository ─────────────────────────────────
    import pandas as pd

    def _empty_ohlcv(*args, **kwargs):
        return None  # triggers fallback in MarketDataService

    def _empty_latest_date(*args, **kwargs):
        return None

    mock_ohlcv_repo = MagicMock()
    mock_ohlcv_repo.return_value.get_price_data.side_effect = _empty_ohlcv
    mock_ohlcv_repo.return_value.get_latest_date.side_effect = _empty_latest_date
    mock_ohlcv_repo.return_value.bulk_upsert.return_value = 0

    monkeypatch.setattr(
        "core.db.repository.OHLCVRepository",
        mock_ohlcv_repo,
    )

    # ── Prediction repository ────────────────────────────
    mock_pred_repo = MagicMock()
    mock_pred_repo.return_value.bulk_insert.return_value = None

    monkeypatch.setattr(
        "core.db.repository.PredictionRepository",
        mock_pred_repo,
    )

    # ── MarketDataService — synthetic price data ─────────
    # Returns a minimal OHLCV DataFrame so any test that
    # instantiates MarketDataService gets usable data without
    # hitting Yahoo Finance or a real DB.

    import numpy as np

    def _synthetic_price_data(ticker="TEST", *args, **kwargs):
        n = 420
        rng = np.random.default_rng(abs(hash(ticker)) % (2**31))
        dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
        price = 100 + rng.normal(0, 1, n).cumsum()
        price = np.abs(price) + 10
        return pd.DataFrame({
            "Open":   price * rng.uniform(0.99, 1.00, n),
            "High":   price * rng.uniform(1.00, 1.01, n),
            "Low":    price * rng.uniform(0.99, 1.00, n),
            "Close":  price,
            "Volume": rng.integers(100_000, 1_000_000, n),
        }, index=dates)

    def _synthetic_batch(tickers, *args, **kwargs):
        price_map = {t: _synthetic_price_data(t) for t in tickers}
        errors = {}
        return price_map, errors

    monkeypatch.setattr(
        "core.data.market_data_service.MarketDataService.get_price_data",
        _synthetic_price_data,
    )
    monkeypatch.setattr(
        "core.data.market_data_service.MarketDataService.get_price_data_batch",
        _synthetic_batch,
    )


# ---------------------------------------------------
# SAFE NETWORK BLOCK
# ---------------------------------------------------

@pytest.fixture(autouse=True)
def disable_external_network(monkeypatch):
    """
    Block external network access while allowing:
    - localhost / 127.0.0.1  (FastAPI TestClient, local postgres)
    - 0.0.0.0                (binding)
    """

    original_socket = socket.socket

    class GuardedSocket(socket.socket):

        def connect(self, address):

            host = address[0]

            if host in ("127.0.0.1", "localhost", "0.0.0.0"):
                return super().connect(address)

            raise RuntimeError(
                f"External network access disabled in test mode: {host}"
            )

    monkeypatch.setattr(socket, "socket", GuardedSocket)

    yield

    monkeypatch.setattr(socket, "socket", original_socket)


# ---------------------------------------------------
# PYTEST GLOBAL CONFIG
# ---------------------------------------------------

def pytest_configure(config):

    os.environ["MARKETSENTINEL_TEST_MODE"] = "1"
    os.environ["SKIP_DATA_SYNC"] = "1"
    os.environ["STORE_PREDICTIONS"] = "0"