import re
import pytest

from core.schema.feature_schema import (
    get_schema_signature,
    CORE_FEATURES,
    CROSS_SECTIONAL_FEATURES,
    SCHEMA_VERSION,
)


# 🔒 IMPORTANT:
# If this test fails — you intentionally changed the model contract.
# You must:
# 1. Bump SCHEMA_VERSION
# 2. Re-train models
# 3. Update EXPECTED_SCHEMA_SIGNATURE below


# ⚠️ Freeze this value AFTER intentional schema update.
# Run once:
# >>> print(get_schema_signature())
# Then paste it here.
EXPECTED_SCHEMA_SIGNATURE = "REPLACE_WITH_REAL_SIGNATURE"


############################################################
# DETERMINISM
############################################################

def test_schema_signature_is_deterministic():
    sig1 = get_schema_signature()
    sig2 = get_schema_signature()
    assert sig1 == sig2


############################################################
# CONTRACT LOCK
############################################################

def test_schema_signature_matches_expected_contract():

    current = get_schema_signature()

    assert current == EXPECTED_SCHEMA_SIGNATURE, (
        "\nSchema signature changed.\n\n"
        "If intentional:\n"
        "- Bump SCHEMA_VERSION\n"
        "- Re-train models\n"
        "- Update EXPECTED_SCHEMA_SIGNATURE in this test\n"
    )


############################################################
# FEATURE SANITY
############################################################

def test_core_features_non_empty():
    assert len(CORE_FEATURES) > 0
    assert all(isinstance(f, str) for f in CORE_FEATURES)


def test_cross_sectional_features_defined():
    assert isinstance(CROSS_SECTIONAL_FEATURES, tuple)
    assert len(CROSS_SECTIONAL_FEATURES) > 0
    assert all(isinstance(f, str) for f in CROSS_SECTIONAL_FEATURES)


def test_no_duplicate_features():
    combined = list(CORE_FEATURES) + list(CROSS_SECTIONAL_FEATURES)
    assert len(combined) == len(set(combined))


############################################################
# VERSION FORMAT
############################################################

def test_schema_version_string_present():
    assert isinstance(SCHEMA_VERSION, str)
    assert len(SCHEMA_VERSION) > 0


def test_schema_version_format():
    # enforce institutional numeric format like "41.0"
    assert re.match(r"^\d+\.\d+$", SCHEMA_VERSION)