"""
Tests for InferencePipeline._filter_latest_per_ticker.
Covers issue #6 — snapshot was returning 27400 rows instead of 100.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from app.inference.pipeline import InferencePipeline

# =====================================================
# HELPERS
# =====================================================


def _make_multi_date_frame(
    tickers: list,
    days: int,
    base_date: str = "2026-01-01",
) -> pd.DataFrame:
    """
    Build a realistic multi-date feature frame.
    Simulates what _build_cross_sectional_frame() returns:
    len(tickers) x days rows.
    """
    rows = []
    base = pd.Timestamp(base_date, tz="UTC")
    for ticker in tickers:
        for i in range(days):
            date = base + timedelta(days=i)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100.0 + np.random.randn(),
                    "rsi_14": 50.0 + np.random.randn(),
                    "raw_model_score": np.random.randn(),
                }
            )
    return pd.DataFrame(rows)


# =====================================================
# CORE FILTER TESTS
# =====================================================


class TestFilterLatestPerTicker:

    def test_reduces_to_one_row_per_ticker(self):
        """
        Regression test for issue #6.
        274 days x 100 tickers = 27400 rows must reduce to 100 rows.
        """
        tickers = [f"TICK{i:03d}" for i in range(100)]
        df = _make_multi_date_frame(tickers, days=274)

        assert len(df) == 27400

        result = InferencePipeline._filter_latest_per_ticker(df)

        assert len(result) == 100
        assert result["ticker"].nunique() == 100

    def test_keeps_latest_date_per_ticker(self):
        """Each ticker must have its latest date row, not earliest."""
        tickers = ["AAPL", "MSFT"]
        df = _make_multi_date_frame(tickers, days=10)

        result = InferencePipeline._filter_latest_per_ticker(df)

        for ticker in tickers:
            ticker_rows = df[df["ticker"] == ticker]
            latest_date = ticker_rows["date"].max()
            result_date = result[result["ticker"] == ticker]["date"].iloc[0]
            assert result_date == latest_date

    def test_handles_empty_dataframe(self):
        """Empty input must return empty output without error."""
        df = pd.DataFrame()
        result = InferencePipeline._filter_latest_per_ticker(df)
        assert result.empty

    def test_handles_none_input(self):
        """None input must return None without error."""
        result = InferencePipeline._filter_latest_per_ticker(None)
        assert result is None

    def test_handles_single_ticker_single_day(self):
        """Single row must pass through unchanged."""
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "date": pd.Timestamp("2026-03-24", tz="UTC"),
                    "close": 200.0,
                }
            ]
        )
        result = InferencePipeline._filter_latest_per_ticker(df)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"

    def test_handles_single_ticker_multiple_days(self):
        """Single ticker with many days must return exactly 1 row."""
        df = _make_multi_date_frame(["AAPL"], days=274)
        result = InferencePipeline._filter_latest_per_ticker(df)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"

    def test_no_duplicate_tickers_in_output(self):
        """Output must never have duplicate tickers."""
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM"]
        df = _make_multi_date_frame(tickers, days=274)

        result = InferencePipeline._filter_latest_per_ticker(df)

        assert result["ticker"].duplicated().sum() == 0

    def test_realistic_universe_100_tickers_274_days(self):
        """
        Full realistic scenario:
        100 tickers x 274 days = 27400 rows → 100 rows.
        This is the exact fix for issue #6.
        """
        tickers = [f"T{i:03d}" for i in range(100)]
        df = _make_multi_date_frame(tickers, days=274)

        assert len(df) == 27400

        result = InferencePipeline._filter_latest_per_ticker(df)

        # Core assertion — this was the bug
        assert len(result) == 100, (
            f"Expected 100 rows (1 per ticker), got {len(result)}. "
            f"Issue #6: snapshot was returning {len(df)} rows."
        )

    def test_handles_mixed_date_formats(self):
        """Date column with string dates must still work."""
        df = pd.DataFrame(
            [
                {"ticker": "AAPL", "date": "2026-01-01", "close": 100.0},
                {"ticker": "AAPL", "date": "2026-03-24", "close": 200.0},
                {"ticker": "MSFT", "date": "2026-01-01", "close": 300.0},
                {"ticker": "MSFT", "date": "2026-03-24", "close": 400.0},
            ]
        )

        result = InferencePipeline._filter_latest_per_ticker(df)

        assert len(result) == 2
        aapl = result[result["ticker"] == "AAPL"].iloc[0]
        assert float(aapl["close"]) == 200.0

    def test_handles_nan_dates_gracefully(self):
        """Rows with NaT/NaN dates must be dropped, not crash."""
        df = pd.DataFrame(
            [
                {"ticker": "AAPL", "date": "2026-03-24", "close": 100.0},
                {"ticker": "AAPL", "date": None, "close": 200.0},
                {"ticker": "MSFT", "date": "2026-03-24", "close": 300.0},
            ]
        )

        result = InferencePipeline._filter_latest_per_ticker(df)

        assert result is not None
        assert "AAPL" in result["ticker"].values
        assert "MSFT" in result["ticker"].values
