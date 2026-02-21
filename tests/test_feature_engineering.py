import pandas as pd
import numpy as np

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# SAMPLE DATA
############################################################

def sample_data(rows=150):

    price_df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=rows),
        "close": np.linspace(100, 120, rows) + np.random.normal(0, 1, rows)
    })

    sentiment_df = pd.DataFrame({
        "date": price_df["date"],
        "avg_sentiment": np.random.randn(rows),
        "news_count": np.random.randint(1, 10, rows),
        "sentiment_std": np.random.rand(rows)
    })

    return price_df, sentiment_df


############################################################
# TRAINING PIPELINE CONTRACT
############################################################

def test_training_pipeline_produces_all_model_features():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    # Must contain all required model features
    assert set(MODEL_FEATURES).issubset(set(df.columns))

    # No NaNs in model features
    assert not df[MODEL_FEATURES].isna().any().any()

    # No infinite values
    arr = df[MODEL_FEATURES].to_numpy()
    assert np.isfinite(arr).all()

    # Basic sanity features
    required_core = ["return", "volatility", "rsi"]
    for col in required_core:
        assert col in df.columns


############################################################
# INFERENCE PIPELINE CONTRACT
############################################################

def test_inference_pipeline_no_training_artifacts():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=False
    )

    assert set(MODEL_FEATURES).issubset(set(df.columns))

    # Inference should not add training-only columns
    assert "target" not in df.columns

    arr = df[MODEL_FEATURES].to_numpy()
    assert np.isfinite(arr).all()


############################################################
# SENTIMENT FALLBACK
############################################################

def test_pipeline_handles_missing_sentiment():

    price, _ = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment_df=None,
        training=True
    )

    assert set(MODEL_FEATURES).issubset(set(df.columns))

    arr = df[MODEL_FEATURES].to_numpy()
    assert np.isfinite(arr).all()


############################################################
# COLUMN STABILITY
############################################################

def test_model_feature_order_stable():

    price, sentiment = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price,
        sentiment,
        training=True
    )

    # Order must match MODEL_FEATURES when selected
    selected = list(df.loc[:, MODEL_FEATURES].columns)

    assert selected == list(MODEL_FEATURES)