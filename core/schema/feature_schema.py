from typing import Tuple, List
import pandas as pd
import hashlib
import numpy as np
import json
import re
import logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "30.0"

############################################################
# CORE FEATURES (TIME-SERIES STABLE ONLY)
############################################################

CORE_FEATURES: Tuple[str, ...] = (

    "return",
    "return_lag1",
    "return_lag5",
    "return_lag10",

    "volatility",
    "volatility_5",
    "volatility_20",

    "momentum_20",

    "rsi",
    "macd",
    "macd_signal",

    "ema_10",
    "ema_50",
    "ema_ratio",

    "regime_feature",
)

############################################################
# CROSS-SECTIONAL FEATURES (OPTIONAL – AUTO DROP)
############################################################

CROSS_SECTIONAL_FEATURES: Tuple[str, ...] = (

    "momentum_20_z",
    "return_lag5_z",
    "rsi_z",
    "volatility_z",
    "ema_ratio_z",

    "momentum_20_rank",
    "return_lag5_rank",
    "rsi_rank",
    "volatility_rank",
    "ema_ratio_rank",
)

############################################################
# MODEL FEATURE CONTRACT (ORDERED, PANDAS-SAFE)
############################################################

MODEL_FEATURES: List[str] = list(
    CORE_FEATURES + CROSS_SECTIONAL_FEATURES
)

DTYPE = np.float32
MIN_ROWS = 200
MIN_VARIANCE = 1e-8

FORBIDDEN_REGEX = re.compile(
    r"\b(future|next|forward|target|label|tomorrow|lead|horizon|lookahead|outcome|response)\b",
    re.IGNORECASE
)


def _check_forbidden_columns(df: pd.DataFrame):
    for col in df.columns:
        if FORBIDDEN_REGEX.search(col):
            raise RuntimeError(f"Lookahead column detected: {col}")


def validate_feature_schema(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        raise RuntimeError("Empty feature dataset.")

    if len(df) < MIN_ROWS:
        raise RuntimeError("Dataset below minimum rows.")

    _check_forbidden_columns(df)

    missing_core = set(CORE_FEATURES) - set(df.columns)
    if missing_core:
        raise RuntimeError(f"Missing core features: {missing_core}")

    feature_df = df.loc[:, [c for c in MODEL_FEATURES if c in df.columns]]
    feature_df = feature_df.astype(DTYPE, copy=True)
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    ########################################################
    # Strict validation on CORE only
    ########################################################

    for col in CORE_FEATURES:

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.nunique() <= 1:
            raise RuntimeError(f"Constant CORE feature detected: {col}")

        if finite_vals.var(ddof=0) < MIN_VARIANCE:
            logger.warning(f"Low variance core feature: {col}")

    ########################################################
    # Auto-drop constant CROSS features
    ########################################################

    drop_cols = []

    for col in CROSS_SECTIONAL_FEATURES:
        if col not in feature_df.columns:
            continue

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.nunique() <= 1:
            logger.warning(f"Dropping constant cross feature: {col}")
            drop_cols.append(col)

    if drop_cols:
        feature_df.drop(columns=drop_cols, inplace=True)

    return feature_df.astype(DTYPE, copy=False)


def get_schema_signature() -> str:

    contract = {
        "core": list(CORE_FEATURES),
        "cross": list(CROSS_SECTIONAL_FEATURES),
        "version": SCHEMA_VERSION,
    }

    canonical = json.dumps(contract, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()