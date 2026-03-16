from typing import Tuple, List, Dict
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

SCHEMA_VERSION = "45.2"


############################################################
# SIGNAL CONTRACT
############################################################

LONG_PERCENTILE = 0.70
SHORT_PERCENTILE = 0.30


############################################################
# CORE FEATURES
############################################################

CORE_FEATURES: Tuple[str, ...] = (

    "return",
    "return_lag1",
    "return_lag5",
    "return_mean_20",
    "reversal_5",

    "momentum_20",
    "momentum_60",
    "momentum_composite",
    "mom_vol_adj",
    "momentum_regime_interaction",

    "volatility",
    "volatility_20",
    "vol_of_vol",
    "return_skew_20",

    "volume_momentum",
    "dollar_volume",
    "amihud",

    "rsi",
    "ema_ratio",

    "dist_from_52w_high",
    "regime_feature",

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
    "mom_vol_adj",
    "momentum_regime_interaction",
    "return_lag5",
    "return_mean_20",
    "reversal_5",
    "rsi",
    "volatility",
    "vol_of_vol",
    "return_skew_20",
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
    re.IGNORECASE,
)


############################################################
# LOG SUPPRESSION
############################################################

_logged_low_variance_core = set()
_logged_low_variance_cs = set()
_logged_constant_cs = set()


############################################################
# INTERNAL UTILITIES
############################################################

def _check_forbidden_columns(df: pd.DataFrame) -> None:

    for col in df.columns:

        if FORBIDDEN_REGEX.search(col):

            raise RuntimeError(f"Lookahead column detected: {col}")


def _safe_numeric_block(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    cols = [c for c in MODEL_FEATURES if c in df.columns]

    for col in cols:

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def _check_dtype_stability(df: pd.DataFrame) -> None:

    numeric_cols = df.select_dtypes(include=["number"]).columns

    bad = [
        c for c in numeric_cols
        if not np.issubdtype(df[c].dtype, np.number)
    ]

    if bad:

        logger.debug(
            "Unexpected dtype detected: %s",
            bad
        )


def _fill_nan_defaults(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:

        if df[col].isna().any():

            if col.endswith("_rank"):
                df[col] = df[col].fillna(0.5)
            else:
                df[col] = df[col].fillna(0.0)

    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:

    return df.loc[:, MODEL_FEATURES]


############################################################
# VALIDATION
############################################################

def validate_feature_schema(
    df: pd.DataFrame,
    mode: str = "training",
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
    # STRICT CONTRACT
    ########################################################

    if mode in {"training", "strict_contract"}:

        missing_all = set(MODEL_FEATURES) - set(feature_df.columns)

        if missing_all:

            raise RuntimeError(
                f"Missing required features under strict contract: {missing_all}"
            )

        feature_df = feature_df.loc[:, MODEL_FEATURES]

    ########################################################
    # INFERENCE MODE
    ########################################################

    elif mode == "inference":

        for col in MODEL_FEATURES:

            if col not in feature_df.columns:

                logger.debug(
                    "Inference auto-created missing feature: %s",
                    col
                )

                if col.endswith("_rank"):
                    feature_df[col] = 0.5
                else:
                    feature_df[col] = 0.0

        feature_df = _reorder_columns(feature_df)

    ########################################################
    # NUMERIC CLEANUP
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

                raise RuntimeError(
                    f"Constant CORE feature detected: {col}"
                )

        if finite_vals.var(ddof=0) < MIN_VARIANCE:

            if col not in _logged_low_variance_core:

                logger.warning(
                    "Low variance core feature: %s",
                    col
                )

                _logged_low_variance_core.add(col)

    ########################################################
    # CROSS SECTIONAL VALIDATION
    ########################################################

    for col in CROSS_SECTIONAL_FEATURES:

        series = feature_df[col]

        finite_vals = series[np.isfinite(series)]

        if len(finite_vals) == 0:
            continue

        if finite_vals.nunique() <= 1:

            if mode == "training":

                if col not in _logged_constant_cs:

                    logger.warning(
                        "Constant CS feature in training: %s",
                        col
                    )

                    _logged_constant_cs.add(col)

                continue

            if col.endswith("_z"):
                feature_df[col] = 0.0
                continue

            if col.endswith("_rank"):
                feature_df[col] = 0.5
                continue

        if finite_vals.var(ddof=0) < MIN_CS_VARIANCE:

            if col not in _logged_low_variance_cs:

                logger.warning(
                    "Low variance CS feature: %s",
                    col
                )

                _logged_low_variance_cs.add(col)

    ########################################################
    # FINAL SANITY
    ########################################################

    if mode == "inference":

        feature_df = _fill_nan_defaults(feature_df)

    else:

        if feature_df.isnull().any().any():

            raise RuntimeError(
                "NaN detected after schema validation."
            )

    if not np.isfinite(feature_df.values).all():

        raise RuntimeError(
            "Non-finite values detected."
        )

    _check_dtype_stability(feature_df)

    feature_df = feature_df.loc[:, MODEL_FEATURES]

    feature_df = feature_df.clip(-1e9, 1e9)

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
        "feature_count": len(MODEL_FEATURES),
    }

    canonical = json.dumps(contract, sort_keys=True)

    return hashlib.sha256(
        canonical.encode()
    ).hexdigest()


############################################################
# SCHEMA SNAPSHOT
############################################################

def schema_snapshot() -> Dict:

    return {
        "version": SCHEMA_VERSION,
        "feature_count": len(MODEL_FEATURES),
        "core_count": len(CORE_FEATURES),
        "cross_count": len(CROSS_SECTIONAL_FEATURES),
        "long_percentile": LONG_PERCENTILE,
        "short_percentile": SHORT_PERCENTILE,
        "signature": get_schema_signature(),
    }