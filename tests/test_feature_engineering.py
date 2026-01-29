def test_feature_columns_created():
    from app.services.feature_engineering import FeatureEngineer
    import pandas as pd

    fe = FeatureEngineer()

    df = pd.DataFrame({
        "close": [100, 102, 101, 103],
        "return": [0.02, -0.01, 0.02, 0.01],
        "avg_sentiment": [0.1, -0.2, 0.3, 0.0]
    })

    df = fe.create_ml_dataset(df)
    assert "target" in df.columns
