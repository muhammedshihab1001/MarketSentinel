from typing import Tuple, Dict
import pandas as pd
import hashlib
import numpy as np
import json


SCHEMA_VERSION = "10.0"


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


FORBIDDEN_PATTERNS = (
    "future",
    "next",
    "tomorrow",
    "target",
    "label",
    "t+",
    "lead",
    "forward"
)


def _check_forbidden_columns(df: pd.DataFrame):

    for col in df.columns:

        if not isinstance(col, str):
            raise RuntimeError("Non-string column detected.")

        tokens = col.lower().replace("+", "_").split("_")

        for bad in FORBIDDEN_PATTERNS:
            if bad in tokens:
                raise RuntimeError(
                    f"Potential lookahead column detected: {col}"
                )


def _build_feature_lock():

    contract = {
        "features": MODEL_FEATURES,
        "dtype": DTYPE,
        "limits": FEATURE_LIMITS,
        "nan_feature": MAX_NAN_RATIO_PER_FEATURE,
        "nan_row": MAX_ROW_NAN_RATIO,
        "abs_limit": ABSOLUTE_FEATURE_LIMIT,
        "variance_floor": MIN_VARIANCE,
        "forbidden": sorted(FORBIDDEN_PATTERNS),
        "count": FEATURE_COUNT,
        "version": SCHEMA_VERSION
    }

    canonical = json.dumps(contract, sort_keys=True)

    return hashlib.sha256(canonical.encode()).hexdigest()


FEATURE_LOCK_HASH = _build_feature_lock()


def validate_feature_schema(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        raise RuntimeError("Feature dataset is empty.")

    if isinstance(df.columns, pd.MultiIndex):
        raise RuntimeError("MultiIndex columns are not allowed.")

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate columns detected.")

    if tuple(df.columns) != MODEL_FEATURES:
        raise RuntimeError("Feature order drift detected.")

    if len(df.columns) != FEATURE_COUNT:
        raise RuntimeError("Feature count mismatch detected.")

    missing_limits = set(MODEL_FEATURES) - set(FEATURE_LIMITS.keys())

    if missing_limits:
        raise RuntimeError(
            f"Missing feature limits for: {missing_limits}"
        )

    _check_forbidden_columns(df)

    feature_df = df.copy(deep=True)

    feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    for col in MODEL_FEATURES:

        coerced = pd.to_numeric(
            feature_df[col],
            errors="coerce"
        )

        if coerced.isna().sum() > feature_df[col].isna().sum():
            raise RuntimeError(
                f"Numeric coercion introduced NaNs in feature: {col}"
            )

        feature_df[col] = coerced.astype(DTYPE)

        finite_vals = feature_df[col][np.isfinite(feature_df[col])]

        if finite_vals.empty:
            raise RuntimeError(
                f"No finite values present in feature: {col}"
            )

        if finite_vals.nunique() <= 1:
            raise RuntimeError(
                f"Constant feature detected: {col}"
            )

        if finite_vals.var() < MIN_VARIANCE:
            raise RuntimeError(
                f"Near-zero variance feature detected: {col}"
            )

        if np.abs(finite_vals).max() > ABSOLUTE_FEATURE_LIMIT:
            raise RuntimeError(
                f"Feature explosion detected: {col}"
            )

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

    for col, (lo, hi) in FEATURE_LIMITS.items():

        series = feature_df[col]
        finite = series[np.isfinite(series)]

        if finite.empty:
            continue

        if (finite < lo).any() or (finite > hi).any():
            raise RuntimeError(
                f"Feature out of plausible bounds: {col}"
            )

    arr = np.ascontiguousarray(
        feature_df.to_numpy(dtype=DTYPE)
    )

    if arr.dtype != np.float32:
        raise RuntimeError("Feature block not float32.")

    return pd.DataFrame(
        arr.copy(),
        columns=MODEL_FEATURES
    )


def get_schema_signature() -> str:

    contract = {
        "lock": FEATURE_LOCK_HASH
    }

    canonical = json.dumps(contract, sort_keys=True)

    return hashlib.sha256(canonical.encode()).hexdigest()
