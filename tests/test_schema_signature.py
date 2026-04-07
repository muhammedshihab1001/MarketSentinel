import os
import re
import pytest

from core.schema.feature_schema import (
    get_schema_signature,
    schema_snapshot,
    CORE_FEATURES,
    CROSS_SECTIONAL_FEATURES,
    SCHEMA_VERSION,
)


############################################################
# CONTRACT LOCK
############################################################

# Optional schema freeze (for model reproducibility).
# For a CV project this can remain optional.

EXPECTED_SCHEMA_SIGNATURE = os.getenv(
    "SCHEMA_EXPECTED_SIGNATURE",
    "REPLACE_WITH_REAL_SIGNATURE"
)


############################################################
# DETERMINISM
############################################################

def test_schema_signature_is_deterministic():

    sig1 = get_schema_signature()
    sig2 = get_schema_signature()

    assert sig1 == sig2


############################################################
# CONTRACT LOCK (OPTIONAL)
############################################################

def test_schema_signature_matches_expected_contract():

    current = get_schema_signature()

    if EXPECTED_SCHEMA_SIGNATURE == "REPLACE_WITH_REAL_SIGNATURE":

        pytest.skip(
            "Schema signature not frozen yet. "
            "Run print(get_schema_signature()) and set it "
            "if you want strict schema locking."
        )

    assert current == EXPECTED_SCHEMA_SIGNATURE, (
        "\nSchema signature changed.\n\n"
        "If intentional:\n"
        "- bump SCHEMA_VERSION\n"
        "- retrain models\n"
        "- update EXPECTED_SCHEMA_SIGNATURE\n"
    )


############################################################
# SIGNATURE FORMAT
############################################################

def test_schema_signature_is_sha256():

    sig = get_schema_signature()

    assert re.match(r"^[a-f0-9]{64}$", sig)


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


def test_no_overlap_between_core_and_cross():

    assert not set(CORE_FEATURES).intersection(
        set(CROSS_SECTIONAL_FEATURES)
    )


def test_cross_sectional_suffixes():

    for f in CROSS_SECTIONAL_FEATURES:
        assert f.endswith("_z") or f.endswith("_rank")


############################################################
# BASIC FEATURE COUNT CHECK
############################################################

def test_total_feature_count_reasonable():

    combined = list(CORE_FEATURES) + list(CROSS_SECTIONAL_FEATURES)

    # sanity range so accidental deletion is caught
    assert 10 < len(combined) < 500


############################################################
# SNAPSHOT CONSISTENCY
############################################################

def test_schema_snapshot_consistent_with_signature():

    snapshot = schema_snapshot()

    assert snapshot["signature"] == get_schema_signature()
    assert snapshot["version"] == SCHEMA_VERSION


############################################################
# VERSION FORMAT
############################################################

def test_schema_version_string_present():

    assert isinstance(SCHEMA_VERSION, str)
    assert len(SCHEMA_VERSION) > 0


def test_schema_version_format():

    # Institutional numeric format like "41.0"
    assert re.match(r"^\d+\.\d+$", SCHEMA_VERSION)
