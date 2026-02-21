import pytest
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import CORE_FEATURES


def test_training_pipeline_produces_all_core_features(sample_data):

    price, sentiment = sample_data

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    assert set(CORE_FEATURES).issubset(set(df.columns))


def test_inference_pipeline_no_training_artifacts(sample_data):

    price, sentiment = sample_data

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=False
    )

    assert set(CORE_FEATURES).issubset(set(df.columns))


def test_pipeline_handles_missing_sentiment(sample_data):

    price, _ = sample_data

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment_df=None,
        training=True
    )

    assert set(CORE_FEATURES).issubset(set(df.columns))


def test_core_feature_order_stable(sample_data):

    price, sentiment = sample_data

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    selected = list(df.loc[:, CORE_FEATURES].columns)

    assert selected == list(CORE_FEATURES)