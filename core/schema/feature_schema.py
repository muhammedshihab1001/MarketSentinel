from typing import List
import pandas as pd


# =====================================================
# SCHEMA VERSION (CRITICAL FOR GOVERNANCE)
# =====================================================

SCHEMA_VERSION = "2.0"


# =====================================================
# CANONICAL FEATURE ORDER
# =====================================================

MODEL_FEATURES: List[str] = [
    "return",
    "volatility",
    "rsi",
    "macd",
    "macd_signal",
    "avg_sentiment",
    "news_count",
    "sentiment_std",
    "return_lag1",
    "sentiment_lag1"
]


# enforce numeric expectation
NUMERIC_FEATURES = set(MODEL_FEATURES)


# =====================================================
# VALIDATION
# =====================================================

def validate_feature_schema(df: pd.DataFrame):
    """
    Institutional schema guard.

    Enforces:

    - column presence
    - no unexpected columns
    - numeric dtype
    - deterministic order
    """

    if df.empty:
        raise RuntimeError("Feature dataset is empty.")

    incoming = list(df.columns)

    missing = set(MODEL_FEATURES) - set(incoming)
    extra = set(incoming) - set(MODEL_FEATURES)

    if missing:
        raise RuntimeError(
            f"Missing required features: {sorted(missing)}"
        )

    if extra:
        raise RuntimeError(
            f"Unexpected features detected: {sorted(extra)}"
        )

    # dtype enforcement
    non_numeric = [
        col for col in MODEL_FEATURES
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise RuntimeError(
            f"Non-numeric features detected: {non_numeric}"
        )

    # enforce deterministic order
    if incoming != MODEL_FEATURES:
        df = df[MODEL_FEATURES]

    return df


# =====================================================
# FINGERPRINT SUPPORT
# =====================================================

def get_schema_signature() -> str:
    """
    Produces deterministic schema fingerprint.

    Used in metadata to prevent
    training/inference skew.
    """

    raw = "|".join(MODEL_FEATURES) + f":{SCHEMA_VERSION}"

    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()
