from typing import Tuple, Dict
import pandas as pd
import hashlib
import numpy as np
import json


SCHEMA_VERSION = "7.1"


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

ABSOLUTE_FEATURE_LIMIT = 1e6


FEATURE_LIMITS: Dict[str, tuple] = {

    "return": (-3.0, 3.0),
    "return_lag1": (-3.0, 3.0),

    "volatility": (0.0, 10.0),

    "rsi": (0.0, 100.0),

    "macd": (-500.0, 500.0),
    "macd_signal": (-500.0, 500.0),

    "avg_sentiment": (-1.0, 1.0),
    "sentiment_lag1": (-1.0, 1.0),

    "news_count": (0, 5000),

    "sentiment_std": (0.0, 10.0),
}


########################################################
# LOOKAHEAD GUARD
########################################################

FORBIDDEN_PATTERNS = (
    "future",
    "next",
    "tomorrow",
    "target",
    "label",
    "t+",
)


def _check_forbidden_columns(df: pd.DataFrame):

    lowered = [c.lower() for c in df.columns]

    for col in lowered:
        for bad in FORBIDDEN_PATTERNS:
            if bad in col:
                raise RuntimeError(
                    f"Potential lookahead column detected: {col}"
                )


########################################################
# VALIDATOR
########################################################

def validate_feature_schema(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        raise RuntimeError("Feature dataset is empty.")

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate columns detected.")

    _check_forbidden_columns(df)

    incoming = set(df.columns)
    expected = set(MODEL_FEATURES)

    missing = expected - incoming
    unknown = incoming - expected

    if missing:
        raise RuntimeError(
            f"Missing required features: {sorted(missing)}"
        )

    if unknown:
        raise RuntimeError(
            f"Unknown features detected: {sorted(unknown)}"
        )

    feature_df = df.loc[:, MODEL_FEATURES].copy(deep=True)

    feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    ########################################################
    # TYPE ENFORCEMENT
    ########################################################

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

        finite_vals = feature_df[col][np.isfinite(feature_df[col])]

        if finite_vals.empty:
            raise RuntimeError(
                f"No finite values present in feature: {col}"
            )

        if np.abs(finite_vals).max() > ABSOLUTE_FEATURE_LIMIT:
            raise RuntimeError(
                f"Feature explosion detected: {col}"
            )

    ########################################################
    # NAN GUARDS
    ########################################################

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

    ########################################################
    # RANGE GUARDS
    ########################################################

    for col, (lo, hi) in FEATURE_LIMITS.items():

        series = feature_df[col]
        finite = series[np.isfinite(series)]

        if finite.empty:
            continue

        if (finite < lo).any() or (finite > hi).any():
            raise RuntimeError(
                f"Feature out of plausible bounds: {col}"
            )

    ########################################################
    # CONTIGUOUS MEMORY
    ########################################################

    arr = np.ascontiguousarray(
        feature_df.to_numpy(dtype=DTYPE)
    )

    return pd.DataFrame(
        arr.copy(),
        columns=MODEL_FEATURES
    )


########################################################
# SCHEMA SIGNATURE
########################################################

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
        f"abs_limit={ABSOLUTE_FEATURE_LIMIT}",
        f"count={len(MODEL_FEATURES)}",
        f"forbidden={','.join(sorted(FORBIDDEN_PATTERNS))}",
        f"version={SCHEMA_VERSION}"
    ]

    raw = "::".join(raw_parts)

    return hashlib.sha256(raw.encode()).hexdigest()
