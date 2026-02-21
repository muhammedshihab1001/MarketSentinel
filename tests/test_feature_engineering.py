import numpy as np
import pandas as pd

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import CORE_FEATURES


def sample_data():
    dates = pd.date_range("2022-01-01", periods=200, tz="UTC")
    price = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 120, 200) + np.random.normal(0, 1, 200),
        "ticker": "TEST"
    })

    sentiment = pd.DataFrame({
        "date": dates,
        "avg_sentiment": np.random.normal(0, 1, 200),
        "news_count": np.random.randint(1, 5, 200),
        "sentiment_std": np.random.random(200)
    })

    return price, sentiment


def test_training_pipeline_produces_all_core_features():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    assert set(CORE_FEATURES).issubset(set(df.columns))


def test_inference_pipeline_no_training_artifacts():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=False
    )

    assert set(CORE_FEATURES).issubset(set(df.columns))


def test_pipeline_handles_missing_sentiment():

    price, _ = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment_df=None,
        training=True
    )

    assert set(CORE_FEATURES).issubset(set(df.columns))


def test_core_feature_order_stable():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    selected = list(df.loc[:, CORE_FEATURES].columns)

    assert selected == list(CORE_FEATURES)