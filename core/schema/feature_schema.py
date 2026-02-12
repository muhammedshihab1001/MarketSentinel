from typing import Tuple, Dict
import pandas as pd
import hashlib
import numpy as np
import json


########################################################
# VERSION
########################################################

SCHEMA_VERSION = "9.0"   # ← bump


########################################################
# FEATURES
########################################################

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

FEATURE_COUNT = len(MODEL_FEATURES)

NUMERIC_FEATURES = frozenset(MODEL_FEATURES)

DTYPE = "float32"


########################################################
# SAFETY LIMITS
########################################################

MAX_NAN_RATIO_PER_FEATURE = 0.05
MAX_ROW_NAN_RATIO = 0.10

ABSOLUTE_FEATURE_LIMIT = 1e6
MIN_VARIANCE = 1e-8


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
# LOOKAHEAD GUARD (UPGRADED)
########################################################

FORBIDDEN_PATTERNS = (
    "future",
    "next",
    "tomorrow",
    "target",
    "label",
    "t+",
    "t1",
    "lead",
    "forward",
    "shift"
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
# FEATURE LOCK (VERY IMPORTANT)
########################################################

def _feature_lock_hash():

    canonical = json.dumps(
        MODEL_FEATURES,
        separators=(",", ":")
    )

    return hashlib.sha256(canonical.encode()).hexdigest()


FEATURE_LOCK_HASH = _feature_lock_hash()


########################################################
# VALIDATOR
########################################################

def validate_feature_schema(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        raise RuntimeError("Feature dataset is empty.")

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate columns detected.")

    ####################################################
    # HARD ORDER + COUNT CHECK
    ####################################################

    if tuple(df.columns) != MODEL_FEATURES:
        raise RuntimeError("Feature order drift detected.")

    if len(df.columns) != FEATURE_COUNT:
        raise RuntimeError("Feature count mismatch detected.")

    _check_forbidden_columns(df)

    feature_df = df.copy(deep=True)

    feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    ####################################################
    # TYPE + NUMERIC ENFORCEMENT
    ####################################################

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

        ################################################
        # CONSTANT FEATURE GUARD
        ################################################

        if finite_vals.nunique() <= 1:
            raise RuntimeError(
                f"Constant feature detected: {col}"
            )

        ################################################
        # VARIANCE FLOOR
        ################################################

        if finite_vals.var() < MIN_VARIANCE:
            raise RuntimeError(
                f"Near-zero variance feature detected: {col}"
            )

        if np.abs(finite_vals).max() > ABSOLUTE_FEATURE_LIMIT:
            raise RuntimeError(
                f"Feature explosion detected: {col}"
            )

    ####################################################
    # NAN GUARDS
    ####################################################

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

    ####################################################
    # RANGE GUARDS
    ####################################################

    for col, (lo, hi) in FEATURE_LIMITS.items():

        series = feature_df[col]
        finite = series[np.isfinite(series)]

        if finite.empty:
            continue

        if (finite < lo).any() or (finite > hi).any():
            raise RuntimeError(
                f"Feature out of plausible bounds: {col}"
            )

    ####################################################
    # CONTIGUOUS FLOAT32 MEMORY
    ####################################################

    arr = np.ascontiguousarray(
        feature_df.to_numpy(dtype=DTYPE)
    )

    if arr.dtype != np.float32:
        raise RuntimeError("Feature block not float32.")

    return pd.DataFrame(
        arr.copy(),
        columns=MODEL_FEATURES
    )


########################################################
# SCHEMA SIGNATURE (UPGRADED)
########################################################

def get_schema_signature() -> str:

    canonical_limits = json.dumps(
        FEATURE_LIMITS,
        sort_keys=True
    )

    raw_parts = [
        FEATURE_LOCK_HASH,   # ⭐ strongest anchor
        "|".join(MODEL_FEATURES),
        f"dtype={DTYPE}",
        f"nan_feature={MAX_NAN_RATIO_PER_FEATURE}",
        f"nan_row={MAX_ROW_NAN_RATIO}",
        f"limits={canonical_limits}",
        f"abs_limit={ABSOLUTE_FEATURE_LIMIT}",
        f"variance_floor={MIN_VARIANCE}",
        f"count={FEATURE_COUNT}",
        f"forbidden={','.join(sorted(FORBIDDEN_PATTERNS))}",
        f"version={SCHEMA_VERSION}"
    ]

    raw = "::".join(raw_parts)

    return hashlib.sha256(raw.encode()).hexdigest()
