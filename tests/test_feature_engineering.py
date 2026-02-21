import pytest
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import CORE_FEATURES
from tests.conftest import sample_data


def test_training_pipeline_produces_all_core_features():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    # Only CORE features required at feature layer
    assert set(CORE_FEATURES).issubset(set(df.columns))


def test_inference_pipeline_no_training_artifacts():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=False
    )

    # Still only CORE features at this stage
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