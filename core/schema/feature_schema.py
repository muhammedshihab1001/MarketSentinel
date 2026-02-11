from typing import Tuple, Dict
import pandas as pd
import hashlib
import numpy as np


SCHEMA_VERSION = "5.0"


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
MAX_ROW_NAN_RATIO = 0.10

DTYPE = "float32"


FEATURE_LIMITS: Dict[str, tuple] = {

    "return": (-0.5, 0.5),
    "return_lag1": (-0.5, 0.5),

    "volatility": (0.0, 5.0),

    "rsi": (0.0, 100.0),

    "macd": (-50.0, 50.0),
    "macd_signal": (-50.0, 50.0),

    "avg_sentiment": (-1.0, 1.0),
    "sentiment_lag1": (-1.0, 1.0),

    "news_count": (0, 500),

    "sentiment_std": (0.0, 5.0),
}


def validate_feature_schema(df: pd.DataFrame) -> pd.DataFrame:

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

    # FORCE deterministic ordering immediately
    feature_df = df.reindex(columns=MODEL_FEATURES).copy()

    # Replace infinities globally
    feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Safe numeric coercion FIRST
    for col in MODEL_FEATURES:

        feature_df[col] = pd.to_numeric(
            feature_df[col],
            errors="coerce"
        )

        if feature_df[col].isna().all():
            raise RuntimeError(
                f"Feature fully NaN after coercion: {col}"
            )

        feature_df[col] = feature_df[col].astype(DTYPE)

    # NaN checks
    per_feature_nan = feature_df.isna().mean()

    unsafe = per_feature_nan[
        per_feature_nan > MAX_NAN_RATIO_PER_FEATURE
    ]

    if not unsafe.empty:
        raise RuntimeError(
            f"Unsafe NaN ratio detected: {unsafe.to_dict()}"
        )

    row_nan_ratio = feature_df.isna().mean(axis=1)

    if (row_nan_ratio > MAX_ROW_NAN_RATIO).any():
        raise RuntimeError(
            "Row-level NaN explosion detected."
        )

    # Range enforcement
    for col, (lo, hi) in FEATURE_LIMITS.items():

        series = feature_df[col].dropna()

        if series.empty:
            continue

        if (series < lo).any() or (series > hi).any():
            raise RuntimeError(
                f"Feature out of bounds: {col}"
            )

    # Ensure contiguous memory (important for numpy / xgboost)
    feature_df = np.ascontiguousarray(feature_df)
    feature_df = pd.DataFrame(
        feature_df,
        columns=MODEL_FEATURES
    )

    return feature_df


def get_schema_signature() -> str:
    """
    Cryptographic schema fingerprint.
    ANY governance change invalidates models.
    """

    raw = "|".join(MODEL_FEATURES)

    raw += f":dtype={DTYPE}"
    raw += f":nan_feature={MAX_NAN_RATIO_PER_FEATURE}"
    raw += f":nan_row={MAX_ROW_NAN_RATIO}"
    raw += f":limits={str(sorted(FEATURE_LIMITS.items()))}"
    raw += f":count={len(MODEL_FEATURES)}"
    raw += f":v={SCHEMA_VERSION}"

    return hashlib.sha256(raw.encode()).hexdigest()
