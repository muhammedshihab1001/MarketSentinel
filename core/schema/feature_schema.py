from typing import Tuple
import pandas as pd
import hashlib


SCHEMA_VERSION = "3.0"


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

    Validates ONLY model features while allowing
    non-feature columns such as:
        date, ticker, target

    Guarantees:
    - strict feature presence
    - dtype enforcement
    - inf protection
    - per-feature NaN limits
    - deterministic ordering
    """

    if df is None or df.empty:
        raise RuntimeError("Feature dataset is empty.")

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate columns detected.")

    incoming = set(df.columns)

    missing = set(MODEL_FEATURES) - incoming

    if missing:
        raise RuntimeError(
            f"Missing required features: {sorted(missing)}"
        )

    feature_df = df.loc[:, MODEL_FEATURES].copy()

    # Replace infinities BEFORE dtype conversion
    feature_df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    for col in MODEL_FEATURES:

        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            raise RuntimeError(
                f"Non-numeric feature detected: {col}"
            )

        feature_df[col] = feature_df[col].astype("float64")

    nan_ratios = feature_df.isna().mean()

    unsafe = nan_ratios[nan_ratios > MAX_NAN_RATIO_PER_FEATURE]

    if not unsafe.empty:
        raise RuntimeError(
            f"Unsafe NaN ratio detected: {unsafe.to_dict()}"
        )

    # deterministic ordering guarantee
    feature_df = feature_df.loc[:, MODEL_FEATURES]

    return feature_df


def get_schema_signature() -> str:
    """
    Deterministic schema fingerprint.
    Changing ANYTHING here intentionally invalidates old models.
    """

    raw = "|".join(MODEL_FEATURES)
    raw += ":dtype=float64"
    raw += f":count={len(MODEL_FEATURES)}"
    raw += f":v={SCHEMA_VERSION}"

    return hashlib.sha256(raw.encode()).hexdigest()
