from typing import Tuple
import pandas as pd
import hashlib


SCHEMA_VERSION = "2.2"


MODEL_FEATURES: Tuple[str, ...] = (
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
)

NUMERIC_FEATURES = frozenset(MODEL_FEATURES)

MAX_NAN_RATIO_PER_FEATURE = 0.05


def validate_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Institutional schema authority.

    Validates ONLY the model feature subset.

    Guarantees:
    - strict feature presence
    - dtype enforcement
    - NaN safety (per feature)
    - deterministic ordering
    """

    if df.empty:
        raise RuntimeError("Feature dataset is empty.")

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate columns detected.")

    incoming = set(df.columns)

    missing = set(MODEL_FEATURES) - incoming

    if missing:
        raise RuntimeError(
            f"Missing required features: {sorted(missing)}"
        )

    # IMPORTANT:
    # Ignore extra columns — they belong to the dataset, not schema.

    feature_df = df.loc[:, MODEL_FEATURES].copy()

    # dtype enforcement
    for col in MODEL_FEATURES:

        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            raise RuntimeError(
                f"Non-numeric feature detected: {col}"
            )

        feature_df[col] = feature_df[col].astype("float64")

    # per-feature NaN safety
    nan_ratios = feature_df.isna().mean()

    unsafe = nan_ratios[nan_ratios > MAX_NAN_RATIO_PER_FEATURE]

    if not unsafe.empty:
        raise RuntimeError(
            f"Unsafe NaN ratio detected: {unsafe.to_dict()}"
        )

    return feature_df


def get_schema_signature() -> str:
    """
    Deterministic schema fingerprint.
    """

    raw = "|".join(MODEL_FEATURES)
    raw += ":dtype=float64"
    raw += f":count={len(MODEL_FEATURES)}"
    raw += f":v={SCHEMA_VERSION}"

    return hashlib.sha256(raw.encode()).hexdigest()
