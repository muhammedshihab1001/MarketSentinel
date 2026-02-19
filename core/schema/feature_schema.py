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

SCHEMA_VERSION = "23.0"  # walk-forward normalized safe limits


############################################################
# FEATURES
############################################################

MODEL_FEATURES: Tuple[str, ...] = (

    "return",
    "return_lag1",
    "return_lag5",
    "return_lag10",

    "volatility",
    "volatility_5",
    "volatility_20",

    "rsi",
    "macd",
    "macd_signal",

    "ema_10",
    "ema_50",
    "ema_ratio",
)

FEATURE_COUNT = len(MODEL_FEATURES)

DTYPE = np.float32
MIN_ROWS = 200


############################################################
# NAN + STABILITY CONTROLS
############################################################

MAX_NAN_RATIO_PER_FEATURE = 0.05
MAX_ROW_NAN_RATIO = 0.10

ABSOLUTE_FEATURE_LIMIT = 1e4  # widened for normalized values
MIN_VARIANCE = 1e-8


############################################################
# HARD FEATURE LIMITS (RELAXED FOR NORMALIZED DATA)
############################################################

FEATURE_LIMITS: Dict[str, tuple] = {

    "return": (-15.0, 15.0),
    "return_lag1": (-15.0, 15.0),
    "return_lag5": (-15.0, 15.0),
    "return_lag10": (-15.0, 15.0),

    "volatility": (0.0, 20.0),
    "volatility_5": (0.0, 20.0),
    "volatility_20": (0.0, 20.0),

    "rsi": (-10.0, 110.0),

    "macd": (-2000.0, 2000.0),
    "macd_signal": (-2000.0, 2000.0),

    "ema_10": (0.0, 1e6),
    "ema_50": (0.0, 1e6),
    "ema_ratio": (-20.0, 20.0),
}


############################################################
# LOOKAHEAD GUARD
############################################################

FORBIDDEN_REGEX = re.compile(
    r"\b(future|next|forward|target|label|tomorrow|lead|horizon|lookahead|outcome|response)\b",
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
# HARD LIMIT ENFORCEMENT
############################################################

def _enforce_feature_limits(df: pd.DataFrame) -> pd.DataFrame:
    for col in MODEL_FEATURES:
        if col not in FEATURE_LIMITS:
            continue

        lo, hi = FEATURE_LIMITS[col]

        if (df[col] < lo).any() or (df[col] > hi).any():
            raise RuntimeError(
                f"Feature limit breach detected in {col}"
            )

    return df


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

    if set(df.columns) != set(MODEL_FEATURES):
        missing = set(MODEL_FEATURES) - set(df.columns)
        extra = set(df.columns) - set(MODEL_FEATURES)

        raise RuntimeError(
            f"Feature mismatch | missing={missing} extra={extra}"
        )

    df = df.loc[:, MODEL_FEATURES]

    _check_forbidden_columns(df)

    try:
        feature_df = df.astype(DTYPE, copy=True)
    except Exception as e:
        raise RuntimeError(f"Non-numeric feature detected → {e}")

    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_df = _enforce_feature_limits(feature_df)

    for col in MODEL_FEATURES:

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.empty:
            raise RuntimeError(f"No finite values present in feature: {col}")

        if finite_vals.nunique() <= 1:
            raise RuntimeError(f"Constant feature detected: {col}")

        var = finite_vals.var(ddof=0)

        if var < MIN_VARIANCE:
            logger.warning(
                "Low variance feature detected: %s | variance=%.12f",
                col,
                var
            )

        if np.abs(finite_vals).max() > ABSOLUTE_FEATURE_LIMIT:
            raise RuntimeError(f"Feature explosion detected: {col}")

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
