from core.features.feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np


def sample_data():

    price_df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=100),
        "close": np.random.rand(100) * 100
    })

    sentiment_df = pd.DataFrame({
        "date": price_df["date"],
        "avg_sentiment": np.random.randn(100),
        "news_count": np.random.randint(1,10,100),
        "sentiment_std": np.random.rand(100)
    })

    return price_df, sentiment_df


def test_training_pipeline_creates_all_features():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    assert set(df.columns) >= set([
        "return",
        "volatility",
        "rsi"
    ])
