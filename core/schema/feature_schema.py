from typing import Tuple, Dict
import pandas as pd
import hashlib
import numpy as np
import json


SCHEMA_VERSION = "6.1"


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

    # allow real market tails
    "return": (-3.0, 3.0),
    "return_lag1": (-3.0, 3.0),

    "volatility": (0.0, 10.0),

    "rsi": (0.0, 100.0),

    "macd": (-500.0, 500.0),
    "macd_signal": (-500.0, 500.0),

    "avg_sentiment": (-1.0, 1.0),
    "sentiment_lag1": (-1.0, 1.0),

    # major fix
    "news_count": (0, 5000),

    "sentiment_std": (0.0, 10.0),
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

    feature_df = df.reindex(columns=MODEL_FEATURES).copy()

    feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

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

    # range guard (not clipping)
    for col, (lo, hi) in FEATURE_LIMITS.items():

        series = feature_df[col].dropna()

        if series.empty:
            continue

        if (series < lo).any() or (series > hi).any():
            raise RuntimeError(
                f"Feature out of plausible bounds: {col}"
            )

    arr = np.ascontiguousarray(
        feature_df.to_numpy(dtype=DTYPE)
    )

    return pd.DataFrame(
        arr,
        columns=MODEL_FEATURES
    )


def get_schema_signature() -> str:

    canonical_limits = json.dumps(
        FEATURE_LIMITS,
        sort_keys=True
    )

    raw_parts = [
        "|".join(MODEL_FEATURES),
        f"dtype={DTYPE}",
        f"nan_feature={MAX_NAN_RATIO_PER_FEATURE}",
        f"nan_row={MAX_ROW_NAN_RATIO}",
        f"limits={canonical_limits}",
        f"count={len(MODEL_FEATURES)}",
        f"version={SCHEMA_VERSION}"
    ]

    raw = "::".join(raw_parts)

    return hashlib.sha256(raw.encode()).hexdigest()
