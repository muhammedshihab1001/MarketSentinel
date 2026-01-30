import pandas as pd
from app.services.feature_engineering import merge_price_sentiment


def test_feature_merge_without_external_calls():
    price_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "close": [100, 102]
    })

    sentiment_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "avg_sentiment": [0.2, -0.1]
    })

    merged = merge_price_sentiment(price_df, sentiment_df)

    assert "avg_sentiment" in merged.columns
    assert not merged.isnull().any().any()
