from typing import Tuple, Dict
import pandas as pd
import hashlib
import numpy as np
import json
import re
import logging


logger = logging.getLogger(__name__)


############################################################
# SCHEMA VERSION
############################################################

SCHEMA_VERSION = "17.0"   # bumped due to feature removal


############################################################
# FEATURES (IMMUTABLE)
############################################################

MODEL_FEATURES: Tuple[str, ...] = (
    "return",
    "volatility",
    "rsi",
    "macd",
    "macd_signal",
    "avg_sentiment",
    "sentiment_std",
    "return_lag1",
    "sentiment_lag1",
)

FEATURE_COUNT = len(MODEL_FEATURES)

DTYPE = np.float32
MIN_ROWS = 120


############################################################
# NAN + STABILITY CONTROLS
############################################################

MAX_NAN_RATIO_PER_FEATURE = 0.05
MAX_ROW_NAN_RATIO = 0.10

ABSOLUTE_FEATURE_LIMIT = 1e3
MIN_VARIANCE = 1e-8


############################################################
# HARD FEATURE LIMITS
############################################################

FEATURE_LIMITS: Dict[str, tuple] = {

    "return": (-3.0, 3.0),
    "return_lag1": (-3.0, 3.0),

    "volatility": (0.0, 10.0),

    "rsi": (0.0, 100.0),

    "macd": (-500.0, 500.0),
    "macd_signal": (-500.0, 500.0),

    "avg_sentiment": (-1.0, 1.0),
    "sentiment_lag1": (-1.0, 1.0),

    "sentiment_std": (0.0, 10.0),
}


############################################################
# LOOKAHEAD GUARD
############################################################

FORBIDDEN_REGEX = re.compile(
    r"(future|next|forward|target|label|tomorrow|t\+|lead|horizon|shift|lookahead|outcome|response|y_)",
    re.IGNORECASE
)


def _check_forbidden_columns(df: pd.DataFrame):

    for col in df.columns:

        if not isinstance(col, str):
            raise RuntimeError("Non-string column detected.")

        if FORBIDDEN_REGEX.search(col):
            raise RuntimeError(
                f"Potential lookahead column detected: {col}"
            )


############################################################
# SCHEMA LOCK
############################################################

def _build_feature_lock():

    contract = {
        "features": list(MODEL_FEATURES),
        "dtype": "float32",
        "limits": FEATURE_LIMITS,
        "nan_feature": MAX_NAN_RATIO_PER_FEATURE,
        "nan_row": MAX_ROW_NAN_RATIO,
        "abs_limit": ABSOLUTE_FEATURE_LIMIT,
        "variance_floor": MIN_VARIANCE,
        "count": FEATURE_COUNT,
        "version": SCHEMA_VERSION,
    }

    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode()).hexdigest()


FEATURE_LOCK_HASH = _build_feature_lock()


############################################################
# MAIN VALIDATOR
############################################################

def validate_feature_schema(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        raise RuntimeError("Feature dataset is empty.")

    if len(df) < MIN_ROWS:
        raise RuntimeError("Dataset below institutional minimum rows.")

    if isinstance(df.columns, pd.MultiIndex):
        raise RuntimeError("MultiIndex columns are not allowed.")

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate columns detected.")

    if tuple(df.columns) != MODEL_FEATURES:
        raise RuntimeError("Feature order drift detected.")

    if len(df.columns) != FEATURE_COUNT:
        raise RuntimeError("Feature count mismatch detected.")

    _check_forbidden_columns(df)

    ########################################################
    # NUMERIC CAST
    ########################################################

    try:
        feature_df = df.astype(DTYPE, copy=True)
    except Exception as e:
        raise RuntimeError(f"Non-numeric feature detected → {e}")

    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    ########################################################
    # FEATURE STABILITY CHECKS
    ########################################################

    for col in MODEL_FEATURES:

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.empty:
            raise RuntimeError(f"No finite values present in feature: {col}")

        if finite_vals.nunique() <= 1:
            raise RuntimeError(f"Constant feature detected: {col}")

        if finite_vals.var(ddof=0) < MIN_VARIANCE:
            raise RuntimeError(f"Near-zero variance feature detected: {col}")

        if np.abs(finite_vals).max() > ABSOLUTE_FEATURE_LIMIT:
            raise RuntimeError(f"Feature explosion detected: {col}")

    ########################################################
    # NAN CHECKS
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
        logger.warning("Row-level NaN spike detected.")

    ########################################################
    # SOFT LIMIT WARNINGS
    ########################################################

    for col, (lo, hi) in FEATURE_LIMITS.items():

        finite = feature_df[col][np.isfinite(feature_df[col])]

        if finite.empty:
            continue

        if (finite < lo).any() or (finite > hi).any():
            logger.warning(
                "Feature outside historical bounds → %s",
                col,
            )

    return feature_df.astype(DTYPE, copy=False)


############################################################
# SIGNATURE
############################################################

def get_schema_signature() -> str:

    contract = {
        "lock": FEATURE_LOCK_HASH,
        "version": SCHEMA_VERSION,
        "count": FEATURE_COUNT,
    }

    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode()).hexdigest()
