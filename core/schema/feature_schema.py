from typing import Tuple
import pandas as pd
import hashlib


# =====================================================
# SCHEMA VERSION
# =====================================================

SCHEMA_VERSION = "2.1"   # bump after hardening


# =====================================================
# IMMUTABLE FEATURE CONTRACT
# =====================================================

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

NUMERIC_FEATURES = set(MODEL_FEATURES)

MAX_NAN_RATIO = 0.05  # institutional safety threshold


# =====================================================
# VALIDATION
# =====================================================

def validate_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Institutional schema authority.

    Enforces:

    - no duplicate columns
    - strict feature set
    - dtype normalization
    - NaN safety
    - deterministic ordering
    """

    if df.empty:
        raise RuntimeError("Feature dataset is empty.")

    # -------------------------------------------------
    # Duplicate column guard
    # -------------------------------------------------

    if df.columns.duplicated().any():
        raise RuntimeError("Duplicate feature columns detected.")

    incoming = set(df.columns)

    missing = set(MODEL_FEATURES) - incoming
    extra = incoming - set(MODEL_FEATURES)

    if missing:
        raise RuntimeError(
            f"Missing required features: {sorted(missing)}"
        )

    if extra:
        raise RuntimeError(
            f"Unexpected features detected: {sorted(extra)}"
        )

    # -------------------------------------------------
    # Enforce order (IN-PLACE)
    # -------------------------------------------------

    df = df.loc[:, MODEL_FEATURES]

    # -------------------------------------------------
    # dtype normalization
    # -------------------------------------------------

    for col in MODEL_FEATURES:

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise RuntimeError(
                f"Non-numeric feature detected: {col}"
            )

        df[col] = df[col].astype("float64")

    # -------------------------------------------------
    # NaN safety
    # -------------------------------------------------

    nan_ratio = df.isna().mean().mean()

    if nan_ratio > MAX_NAN_RATIO:
        raise RuntimeError(
            f"Feature NaN ratio unsafe: {nan_ratio:.3f}"
        )

    return df


# =====================================================
# SCHEMA SIGNATURE
# =====================================================

def get_schema_signature() -> str:
    """
    Deterministic schema fingerprint.

    Protects against:
    - feature drift
    - dtype drift
    - ordering skew
    """

    raw = "|".join(MODEL_FEATURES)
    raw += f":dtype=float64"
    raw += f":count={len(MODEL_FEATURES)}"
    raw += f":v={SCHEMA_VERSION}"

    return hashlib.sha256(raw.encode()).hexdigest()
