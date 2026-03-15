import numpy as np
import pandas as pd

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# SAMPLE DATA
############################################################

def sample_data(n_days=300):

    np.random.seed(42)

    dates = pd.date_range("2022-01-01", periods=n_days, tz="UTC")

    df = pd.DataFrame({
        "date": np.tile(dates, 3),
        "ticker": np.repeat(["A", "B", "C"], n_days),
        "close": np.random.normal(100, 5, n_days * 3),
        "volume": np.random.randint(1000, 5000, n_days * 3)
    })

    return df


############################################################
# CORE FEATURE GENERATION
############################################################

def test_pipeline_produces_model_features():

    price_df = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    # ensure all model features exist
    assert set(MODEL_FEATURES).issubset(df.columns)


############################################################
# CROSS-SECTIONAL FEATURES EXIST
############################################################

def test_cross_sectional_features_present():

    price_df = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    expected = [
        "market_dispersion",
        "breadth",
        "regime_feature"
    ]

    for col in expected:
        assert col in df.columns


############################################################
# Z-SCORE FEATURES GENERATED
############################################################

def test_zscore_features_exist():

    price_df = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    z_cols = [c for c in df.columns if c.endswith("_z")]

    assert len(z_cols) > 0


############################################################
# RANK FEATURES GENERATED
############################################################

def test_rank_features_exist():

    price_df = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    rank_cols = [c for c in df.columns if c.endswith("_rank")]

    assert len(rank_cols) > 0


############################################################
# VOLATILITY FLOOR ENFORCED
############################################################

def test_volatility_floor():

    price_df = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    assert (df["volatility"] > 0).all()


############################################################
# NON-FINITE GUARD
############################################################

def test_no_infinite_values():

    price_df = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    numeric = df.select_dtypes(include=[np.number])

    assert np.isfinite(numeric.to_numpy()).all()


############################################################
# DATETIME NORMALIZATION
############################################################

def test_datetime_normalization():

    price_df = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    assert str(df["date"].dtype) == "datetime64[ns]"


############################################################
# MODEL FEATURE ORDER STABILITY
############################################################

def test_model_feature_order_stable():

    price_df = sample_data()

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    selected = list(df.loc[:, MODEL_FEATURES].columns)

    assert selected == list(MODEL_FEATURES)


############################################################
# BASIC DATA ROBUSTNESS (YFINANCE STYLE NOISY DATA)
############################################################

def test_pipeline_handles_noisy_data():

    price_df = sample_data()

    # introduce noise
    price_df.loc[10:20, "close"] = np.nan
    price_df.loc[30:40, "volume"] = 0

    df = FeatureEngineer.build_feature_pipeline(
        price_df,
        sentiment_df=None,
        training=True
    )

    assert len(df) > 0