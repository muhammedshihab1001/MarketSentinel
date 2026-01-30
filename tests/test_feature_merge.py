import pandas as pd
from app.services.feature_engineering import FeatureEngineer


def test_feature_merge_without_external_calls():
    price_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "close": [100, 102]
    })

    sentiment_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "avg_sentiment": [0.2, -0.1],
        "news_count": [5, 3],
        "sentiment_std": [0.1, 0.2]
    })

    merged = FeatureEngineer.merge_price_sentiment(price_df, sentiment_df)

    assert "avg_sentiment" in merged.columns
    assert "news_count" in merged.columns
    assert "sentiment_std" in merged.columns
    assert not merged.isnull().any().any()
