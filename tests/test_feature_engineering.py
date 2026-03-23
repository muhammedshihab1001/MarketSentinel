import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# SAMPLE DATA
# FIX (item 41): Use relative dates — was hardcoded 2022-01-01
# (4 years back). Now uses today - 400 days so tests always
# use recent-ish historical dates without drifting stale.
############################################################

def sample_data(n_days=300):

    np.random.seed(42)

    # FIX: relative start — always n_days + 100 days before today
    start = (datetime.utcnow() - timedelta(days=n_days + 100)).date()
    dates = pd.date_range(start=start, periods=n_days, tz="UTC")

    df = pd.DataFrame({
        "date": np.tile(dates, 3),
        "ticker": np.repeat(["A", "B", "C"], n_days),
        "close": np.random.normal(100, 5, n_days * 3),
        "volume": np.random.randint(1000, 5000, n_days * 3),
    })

    return df


############################################################
# CORE FEATURE GENERATION
############################################################

def test_pipeline_produces_model_features():

    price_df = sample_data()
    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)
    assert set(MODEL_FEATURES).issubset(df.columns)


############################################################
# CROSS-SECTIONAL FEATURES EXIST
############################################################

def test_cross_sectional_features_present():

    price_df = sample_data()
    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)

    expected = ["market_dispersion", "breadth", "regime_feature"]

    for col in expected:
        assert col in df.columns


############################################################
# Z-SCORE FEATURES GENERATED
############################################################

def test_zscore_features_exist():

    price_df = sample_data()
    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)
    z_cols = [c for c in df.columns if c.endswith("_z")]
    assert len(z_cols) > 0


############################################################
# RANK FEATURES GENERATED
############################################################

def test_rank_features_exist():

    price_df = sample_data()
    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    assert len(rank_cols) > 0


############################################################
# VOLATILITY FLOOR ENFORCED
############################################################

def test_volatility_floor():

    price_df = sample_data()
    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)
    assert (df["volatility"] > 0).all()


############################################################
# NON-FINITE GUARD
############################################################

def test_no_infinite_values():

    price_df = sample_data()
    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)
    numeric = df.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


############################################################
# DATETIME NORMALIZATION
############################################################

def test_datetime_normalization():

    price_df = sample_data()
    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)
    assert "datetime64" in str(df["date"].dtype)


############################################################
# MODEL FEATURE ORDER STABILITY
############################################################

def test_model_feature_order_stable():

    price_df = sample_data()
    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)
    selected = list(df.loc[:, MODEL_FEATURES].columns)
    assert selected == list(MODEL_FEATURES)


############################################################
# BASIC DATA ROBUSTNESS (YFINANCE STYLE NOISY DATA)
############################################################

def test_pipeline_handles_noisy_data():

    price_df = sample_data()
    price_df.loc[10:20, "close"] = np.nan
    price_df.loc[30:40, "volume"] = 0

    df = FeatureEngineer.build_feature_pipeline(price_df, training=True)

    assert isinstance(df, pd.DataFrame)
    assert set(MODEL_FEATURES).issubset(df.columns)