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

SCHEMA_VERSION = "28.0"  # dynamic optional sentiment handling


############################################################
# CORE FEATURES (MANDATORY – PRICE BASED)
############################################################

CORE_FEATURES: Tuple[str, ...] = (

    # returns
    "return",
    "return_lag1",
    "return_lag5",
    "return_lag10",

    # volatility
    "volatility",
    "volatility_5",
    "volatility_20",

    # momentum
    "momentum_20",

    # technical
    "rsi",
    "macd",
    "macd_signal",

    # trend
    "ema_10",
    "ema_50",
    "ema_ratio",

    # regime
    "regime_feature",
)

############################################################
# OPTIONAL FEATURES (AUTO-DROP IF CONSTANT)
############################################################

OPTIONAL_FEATURES: Tuple[str, ...] = (

    "avg_sentiment",
    "sentiment_std",
    "news_count",
    "sentiment_ema_3",
    "sentiment_momentum",
)

MODEL_FEATURES: Tuple[str, ...] = CORE_FEATURES + OPTIONAL_FEATURES

FEATURE_COUNT = len(MODEL_FEATURES)

DTYPE = np.float32
MIN_ROWS = 200

MAX_NAN_RATIO_PER_FEATURE = 0.05
ABSOLUTE_FEATURE_LIMIT = 1e4
MIN_VARIANCE = 1e-8


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
# VALIDATOR
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

    _check_forbidden_columns(df)

    # Ensure all CORE features exist
    missing_core = set(CORE_FEATURES) - set(df.columns)
    if missing_core:
        raise RuntimeError(f"Missing core features: {missing_core}")

    # Work only on declared model features
    available_features = [
        col for col in MODEL_FEATURES if col in df.columns
    ]

    feature_df = df.loc[:, available_features].astype(DTYPE, copy=True)
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    ########################################################
    # Drop constant OPTIONAL features automatically
    ########################################################

    drop_optional = []

    for col in OPTIONAL_FEATURES:
        if col not in feature_df.columns:
            continue

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.nunique() <= 1:
            logger.warning(
                "Dropping constant optional feature: %s",
                col
            )
            drop_optional.append(col)

    if drop_optional:
        feature_df.drop(columns=drop_optional, inplace=True)

    ########################################################
    # Strict validation on CORE features
    ########################################################

    for col in CORE_FEATURES:

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.empty:
            raise RuntimeError(f"No finite values in feature: {col}")

        if finite_vals.nunique() <= 1:
            raise RuntimeError(f"Constant CORE feature detected: {col}")

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

    return feature_df.astype(DTYPE, copy=False)


############################################################
# SIGNATURE
############################################################

def get_schema_signature() -> str:

    contract = {
        "core": list(CORE_FEATURES),
        "optional": list(OPTIONAL_FEATURES),
        "version": SCHEMA_VERSION,
    }

    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode()).hexdigest()