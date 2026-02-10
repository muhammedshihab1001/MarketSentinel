from core.features.feature_engineering import FeatureEngineer


class FeatureStore:
    """
    Institutional Feature Boundary.

    Today:
        computes features on demand.

    Tomorrow:
        loads precomputed offline features.

    This prevents inference from being tightly coupled
    to feature engineering logic.
    """

    def __init__(self):
        self.engineer = FeatureEngineer()

    # --------------------------------------------------

    def get_features(self, price_df, sentiment_df):
        """
        ZERO behavior change wrapper.

        Future upgrade path:
        - offline parquet features
        - Redis feature cache
        - Feast
        - vector store
        """

        return self.engineer.build_feature_pipeline(
            price_df,
            sentiment_df
        )
