import pandas as pd
import numpy as np

from core.features.feature_engineering import FeatureEngineer


############################################################
# HELPERS
############################################################

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


############################################################
# LEFT JOIN SAFETY
############################################################

def test_merge_preserves_all_price_rows():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    assert len(merged) == 5
    assert set(["close"]).issubset(set(merged.columns))


############################################################
# ZERO FILL GUARANTEE
############################################################

def test_missing_sentiment_is_zero_filled():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    # Missing dates should be zero
    assert (merged["avg_sentiment"].fillna(0) == merged["avg_sentiment"]).all()
    assert (merged["news_count"].fillna(0) == merged["news_count"]).all()
    assert (merged["sentiment_std"].fillna(0) == merged["sentiment_std"]).all()


############################################################
# ORDERING GUARANTEE
############################################################

def test_merge_is_time_sorted():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price().sample(frac=1),
        build_sentiment_partial().sample(frac=1)
    )

    assert merged["date"].is_monotonic_increasing


############################################################
# NUMERIC GUARANTEE
############################################################

def test_sentiment_columns_are_numeric():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    assert pd.api.types.is_numeric_dtype(merged["avg_sentiment"])
    assert pd.api.types.is_numeric_dtype(merged["news_count"])
    assert pd.api.types.is_numeric_dtype(merged["sentiment_std"])


############################################################
# NO DUPLICATES
############################################################

def test_no_duplicate_dates_after_merge():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    assert merged["date"].duplicated().sum() == 0


############################################################
# NO NAN GUARANTEE
############################################################

def test_no_nan_after_merge():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        build_sentiment_partial()
    )

    assert not merged.isna().any().any()


############################################################
# EMPTY SENTIMENT SAFETY
############################################################

def test_empty_sentiment_dataframe():

    merged = FeatureEngineer.merge_price_sentiment(
        build_price(),
        pd.DataFrame(columns=[
            "date", "avg_sentiment", "news_count", "sentiment_std"
        ])
    )

    assert len(merged) == 5
    assert (merged["avg_sentiment"] == 0).all()
    assert (merged["news_count"] == 0).all()
    assert (merged["sentiment_std"] == 0).all()