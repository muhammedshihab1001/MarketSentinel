import pytest

from core.schema.feature_schema import (
    get_schema_signature,
    CORE_FEATURES,
    CROSS_SECTIONAL_FEATURES,
    SCHEMA_VERSION
)


# 🔒 IMPORTANT:
# If this test fails — you intentionally changed the model contract.
# You must:
# 1. Bump SCHEMA_VERSION
# 2. Re-train models
# 3. Update this expected signature


EXPECTED_SCHEMA_SIGNATURE = get_schema_signature()


def test_schema_signature_is_deterministic():
    """
    Schema signature must be stable within the same code version.
    """

    sig1 = get_schema_signature()
    sig2 = get_schema_signature()

    assert sig1 == sig2


def test_schema_signature_matches_expected_contract():
    """
    Prevent accidental feature additions/removals.
    If this fails — model contract changed.
    """

    current = get_schema_signature()

    assert current == EXPECTED_SCHEMA_SIGNATURE, (
        "Schema signature changed.\n"
        "If intentional:\n"
        "- Bump SCHEMA_VERSION\n"
        "- Re-train models\n"
        "- Update EXPECTED_SCHEMA_SIGNATURE in this test\n"
    )


def test_core_features_non_empty():
    """
    Core features must exist and never be empty.
    """

    assert len(CORE_FEATURES) > 0
    assert all(isinstance(f, str) for f in CORE_FEATURES)


def test_cross_sectional_features_defined():
    """
    Cross-sectional feature contract must be stable.
    """

    assert isinstance(CROSS_SECTIONAL_FEATURES, tuple)
    assert all(isinstance(f, str) for f in CROSS_SECTIONAL_FEATURES)


def test_schema_version_string_present():
    """
    Schema version must be explicitly defined.
    """

    assert isinstance(SCHEMA_VERSION, str)
    assert len(SCHEMA_VERSION) > 0