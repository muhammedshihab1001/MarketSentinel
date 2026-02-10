import pandas as pd
import numpy as np

from core.features.feature_engineering import FeatureEngineer


def build_price():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "close": [100, 101, 102, 103, 104]
    })


def build_sentiment_partial():
    # intentionally missing some dates
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-03"
        ]),
        "avg_sentiment": [0.2, -0.1],
        "news_count": [5, 2],
        "sentiment_std": [0.05, 0.2]
    })


# ---------------------------------------------------
# LEFT JOIN SAFETY
# ---------------------------------------------------

def test_merge_preserves_all_price_rows():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    assert len(merged) == 5


# ---------------------------------------------------
# ZERO FILL GUARANTEE
# ---------------------------------------------------

def test_missing_sentiment_is_zero_filled():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    filled_rows = merged[merged["avg_sentiment"] == 0.0]

    assert len(filled_rows) >= 1


# ---------------------------------------------------
# ORDERING GUARANTEE
# ---------------------------------------------------

def test_merge_is_time_sorted():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price().sample(frac=1),
        build_sentiment_partial().sample(frac=1)
    )

    assert merged["date"].is_monotonic_increasing


# ---------------------------------------------------
# NUMERIC GUARANTEE
# ---------------------------------------------------

def test_sentiment_columns_are_numeric():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    numeric = merged.select_dtypes(include="number")

    assert "avg_sentiment" in numeric.columns
    assert "news_count" in numeric.columns
    assert "sentiment_std" in numeric.columns


# ---------------------------------------------------
# NO DUPLICATES
# ---------------------------------------------------

def test_no_duplicate_dates_after_merge():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    assert merged["date"].duplicated().sum() == 0
