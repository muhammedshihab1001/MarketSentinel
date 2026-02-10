import pandas as pd
import numpy as np

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES


def build_price_df(rows=300):
    dates = pd.date_range("2020-01-01", periods=rows)

    return pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 150, rows) + np.random.randn(rows)
    })


def build_sentiment_df(rows=300):
    dates = pd.date_range("2020-01-01", periods=rows)

    return pd.DataFrame({
        "date": dates,
        "avg_sentiment": np.random.randn(rows) * 0.1,
        "news_count": np.random.randint(1, 10, rows),
        "sentiment_std": np.random.rand(rows)
    })


# ---------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------

def test_training_pipeline_creates_all_features():

    fe = FeatureEngineer()

    price = build_price_df()
    sentiment = build_sentiment_df()

    df = fe.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    assert set(df.columns) == set(MODEL_FEATURES + ["target"])


# ---------------------------------------------------
# INFERENCE PIPELINE
# ---------------------------------------------------

def test_inference_pipeline_has_no_target():

    fe = FeatureEngineer()

    price = build_price_df()
    sentiment = build_sentiment_df()

    df = fe.build_feature_pipeline(
        price,
        sentiment,
        training=False
    )

    assert "target" not in df.columns
    assert set(df.columns) == set(MODEL_FEATURES)


# ---------------------------------------------------
# NUMERIC SAFETY
# ---------------------------------------------------

def test_features_are_numeric():

    fe = FeatureEngineer()

    df = fe.build_feature_pipeline(
        build_price_df(),
        build_sentiment_df(),
        training=False
    )

    numeric = df.select_dtypes(include="number")

    assert len(numeric.columns) == len(MODEL_FEATURES)


# ---------------------------------------------------
# NO EMPTY OUTPUT
# ---------------------------------------------------

def test_feature_pipeline_never_returns_empty():

    fe = FeatureEngineer()

    df = fe.build_feature_pipeline(
        build_price_df(400),
        build_sentiment_df(400),
        training=True
    )

    assert len(df) > 50
