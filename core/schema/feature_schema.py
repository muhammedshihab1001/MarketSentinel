from typing import Tuple, List
import pandas as pd
import hashlib
import numpy as np
import json
import re
import logging

logger = logging.getLogger(__name__)

# ============================================================
# SCHEMA VERSION
# ============================================================

SCHEMA_VERSION = "41.0"  # institutional market structure expansion


############################################################
# SIGNAL CONTRACT
############################################################

LONG_PERCENTILE = 0.70
SHORT_PERCENTILE = 0.30


############################################################
# CORE FEATURES (EXPANDED)
############################################################

CORE_FEATURES: Tuple[str, ...] = (

    # Returns
    "return",
    "return_lag1",
    "return_lag5",
    "return_mean_20",

    # Momentum
    "momentum_20",
    "momentum_60",
    "momentum_composite",

    # Volatility
    "volatility",
    "volatility_20",

    # Liquidity
    "volume_momentum",
    "dollar_volume",
    "amihud",

    # Technical
    "rsi",
    "ema_ratio",

    # Structure
    "dist_from_52w_high",
    "regime_feature",

    # Market-level context
    "market_dispersion",
    "breadth",
)


############################################################
# CROSS-SECTIONAL BASE COLUMNS
############################################################

BASE_CS_COLS: Tuple[str, ...] = (
    "momentum_20",
    "momentum_60",
    "momentum_composite",
    "return_lag5",
    "return_mean_20",
    "rsi",
    "volatility",
    "ema_ratio",
    "volume_momentum",
    "dollar_volume",
    "dist_from_52w_high",
    "regime_feature",
    "amihud",
    "market_dispersion",
    "breadth",
)

CROSS_SECTIONAL_FEATURES: Tuple[str, ...] = tuple(
    [f"{col}_z" for col in BASE_CS_COLS] +
    [f"{col}_rank" for col in BASE_CS_COLS]
)


############################################################
# MODEL FEATURE CONTRACT
############################################################

MODEL_FEATURES: List[str] = list(
    CORE_FEATURES + CROSS_SECTIONAL_FEATURES
)

DTYPE = np.float32
MIN_ROWS_TRAINING = 300
MIN_VARIANCE = 1e-8
MIN_CS_VARIANCE = 1e-6

FORBIDDEN_REGEX = re.compile(
    r"\b(future|next|forward|target|label|tomorrow|lead|horizon|lookahead|outcome|response)\b",
    re.IGNORECASE
)


############################################################
# INTERNAL UTILITIES
############################################################

def _check_forbidden_columns(df: pd.DataFrame):
    for col in df.columns:
        if FORBIDDEN_REGEX.search(col):
            raise RuntimeError(f"Lookahead column detected: {col}")


def _safe_numeric_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


############################################################
# VALIDATION
############################################################

def validate_feature_schema(
    df: pd.DataFrame,
    mode: str = "training"
) -> pd.DataFrame:

    if df is None or df.empty:
        raise RuntimeError("Empty feature dataset.")

    if mode not in {"training", "inference", "strict_contract"}:
        raise RuntimeError(f"Unknown validation mode: {mode}")

    if mode == "training" and len(df) < MIN_ROWS_TRAINING:
        raise RuntimeError("Dataset below minimum rows for training.")

    _check_forbidden_columns(df)

    missing_core = set(CORE_FEATURES) - set(df.columns)
    if missing_core:
        raise RuntimeError(f"Missing core features: {missing_core}")

    feature_df = df.copy()

    ########################################################
    # STRICT CONTRACT ENFORCEMENT
    ########################################################

    if mode in {"training", "strict_contract"}:

        missing_all = set(MODEL_FEATURES) - set(feature_df.columns)
        if missing_all:
            raise RuntimeError(
                f"Missing required features under strict contract: {missing_all}"
            )

        feature_df = feature_df.loc[:, MODEL_FEATURES]

    elif mode == "inference":

        for col in MODEL_FEATURES:
            if col not in feature_df.columns:
                if col.endswith("_rank"):
                    feature_df[col] = 0.5
                else:
                    feature_df[col] = 0.0

        feature_df = feature_df.loc[:, MODEL_FEATURES]

    ########################################################
    # NUMERIC + CLEANUP
    ########################################################

    feature_df = _safe_numeric_block(feature_df)

    ########################################################
    # CORE VALIDATION
    ########################################################

    for col in CORE_FEATURES:

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.empty:
            raise RuntimeError(f"Core feature fully invalid: {col}")

        if mode in {"training", "strict_contract"}:
            if finite_vals.nunique() <= 1:
                raise RuntimeError(f"Constant CORE feature detected: {col}")

        if finite_vals.var(ddof=0) < MIN_VARIANCE:
            logger.warning(f"Low variance core feature: {col}")

    ########################################################
    # CROSS-SECTIONAL VALIDATION
    ########################################################

    for col in CROSS_SECTIONAL_FEATURES:

        series = feature_df[col]
        finite_vals = series[np.isfinite(series)]

        if finite_vals.nunique() <= 1:

            if mode == "training":
                logger.warning("Constant CS feature in training: %s", col)
                continue

            if col.endswith("_z"):
                feature_df[col] = 0.0
                continue

            if col.endswith("_rank"):
                feature_df[col] = 0.5
                continue

        if finite_vals.var(ddof=0) < MIN_CS_VARIANCE:
            logger.warning("Low variance CS feature: %s", col)

    ########################################################
    # FINAL SANITY
    ########################################################

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
        "signal": {
            "long_percentile": LONG_PERCENTILE,
            "short_percentile": SHORT_PERCENTILE,
        },
        "version": SCHEMA_VERSION,
    }

    canonical = json.dumps(contract, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()