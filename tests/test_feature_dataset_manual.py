from app.services.feature_engineering import FeatureEngineer

fe = FeatureEngineer()

# Assume df is already merged & indicators added
# Just sanity check column creation

import pandas as pd

df = pd.DataFrame({
    "return": [0.01, -0.02, 0.03],
    "avg_sentiment": [0.2, -0.1, 0.3]
})

df = fe.create_ml_dataset(df)
print(df)
