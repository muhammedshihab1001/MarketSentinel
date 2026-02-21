from typing import Tuple, List
import pandas as pd
import hashlib
import numpy as np
import json
import re
import logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "30.3"  # bumped – inference compatibility fix

############################################################
# CORE FEATURES
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
# CROSS-SECTIONAL FEATURES
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
# MODEL FEATURE CONTRACT (IMMUTABLE)
############################################################

MODEL_FEATURES: List[str] = list(
    CORE_FEATURES + CROSS_SECTIONAL_FEATURES
)

DTYPE = np.float32
MIN_ROWS_TRAINING = 200
MIN_VARIANCE = 1e-8

FORBIDDEN_REGEX = re.compile(
    r"\b(future|next|forward|target|label|tomorrow|lead|horizon|lookahead|outcome|response)\b",
    re.IGNORECASE
)


def _check_forbidden_columns(df: pd.DataFrame):
    for col in df.columns:
        if FORBIDDEN_REGEX.search(col):
            raise RuntimeError(f"Lookahead column detected: {col}")


############################################################
# VALIDATION
############################################################

def validate_feature_schema(
    df: pd.DataFrame,
    mode: str = "training"  # training | inference | strict_contract
) -> pd.DataFrame:

    if df is None or df.empty:
        raise RuntimeError("Empty feature dataset.")

    if mode not in {"training", "inference", "strict_contract"}:
        raise RuntimeError(f"Unknown validation mode: {mode}")

    # -------------------------------------------------------
    # Row requirement (TRAINING ONLY)
    # -------------------------------------------------------
    if mode == "training" and len(df) < MIN_ROWS_TRAINING:
        raise RuntimeError("Dataset below minimum rows for training.")

    _check_forbidden_columns(df)

    missing_core = set(CORE_FEATURES) - set(df.columns)
    if missing_core:
        raise RuntimeError(f"Missing core features: {missing_core}")

    # -------------------------------------------------------
    # STRICT CONTRACT MODE (final training enforcement)
    # -------------------------------------------------------
    if mode == "strict_contract":

        missing_all = set(MODEL_FEATURES) - set(df.columns)
        if missing_all:
            raise RuntimeError(
                f"Strict contract: missing required features {missing_all}"
            )

        feature_df = df.loc[:, MODEL_FEATURES].copy()

    else:
        feature_df = df.loc[
            :, [c for c in MODEL_FEATURES if c in df.columns]
        ].copy()

    feature_df = feature_df.astype(DTYPE)
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -------------------------------------------------------
    # CORE VALIDATION
    # -------------------------------------------------------
    for col in CORE_FEATURES:

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.empty:
            raise RuntimeError(f"Core feature fully invalid: {col}")

        if mode != "inference" and finite_vals.nunique() <= 1:
            raise RuntimeError(f"Constant CORE feature detected: {col}")

        if finite_vals.var(ddof=0) < MIN_VARIANCE:
            logger.warning(f"Low variance core feature: {col}")

    # -------------------------------------------------------
    # CROSS FEATURE BEHAVIOR (non-strict modes)
    # -------------------------------------------------------
    if mode != "strict_contract":

        drop_cols = []

        for col in CROSS_SECTIONAL_FEATURES:

            if col not in feature_df.columns:
                continue

            series = feature_df[col]
            finite_vals = series[np.isfinite(series)]

            if finite_vals.empty or finite_vals.nunique() <= 1:
                logger.warning(
                    f"Dropping constant cross feature: {col}"
                )
                drop_cols.append(col)

        if drop_cols:
            feature_df.drop(columns=drop_cols, inplace=True)

    # -------------------------------------------------------
    # FINAL SANITY CHECK
    # -------------------------------------------------------
    if feature_df.isnull().any().any():
        raise RuntimeError("NaN detected after schema validation.")

    if not np.isfinite(feature_df.values).all():
        raise RuntimeError("Non-finite values detected.")

    return feature_df.astype(DTYPE, copy=False)


############################################################
# SCHEMA SIGNATURE
############################################################

def get_schema_signature() -> str:

    contract = {
        "core": list(CORE_FEATURES),
        "cross": list(CROSS_SECTIONAL_FEATURES),
        "version": SCHEMA_VERSION,
    }

    canonical = json.dumps(contract, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()