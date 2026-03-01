import json
import tempfile
import os
import pytest

from core.artifacts.metadata_manager import MetadataManager


def _build_valid_metadata():

    return MetadataManager.create_metadata(
        model_name="xgboost",
        metrics={"sharpe": 1.2},
        features=tuple(["f1"] * 10),
        training_start="2025-01-01",
        training_end="2026-01-01",
        dataset_hash="abc",
        dataset_rows=1000,
        metadata_type="training_manifest_v1",
        feature_checksum="dummy_checksum",
        extra_fields={
            "artifact_hash": "artifact_hash",
            "schema_signature": "sig",
            "schema_version": "1.0",
            "training_code_hash": "codehash",
            "universe_hash": "uhash"
        }
    )


# ======================================================
# BASIC INTEGRITY
# ======================================================

def test_metadata_integrity_roundtrip():

    metadata = _build_valid_metadata()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        MetadataManager.save_metadata(metadata, tmp.name)
        loaded = MetadataManager.load_metadata(tmp.name)

    assert loaded["metadata_integrity_hash"] == metadata["metadata_integrity_hash"]

    os.remove(tmp.name)


# ======================================================
# REQUIRED FIELDS PRESENT
# ======================================================

def test_metadata_required_fields_present():

    metadata = _build_valid_metadata()

    required_keys = {
        "metadata_integrity_hash",
        "metadata_type",
        "dataset_hash",
        "dataset_rows",
        "features",
        "training_start",
        "training_end"
    }

    assert required_keys.issubset(metadata.keys())


# ======================================================
# TAMPER DETECTION
# ======================================================

def test_metadata_tamper_detection():

    metadata = _build_valid_metadata()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        MetadataManager.save_metadata(metadata, tmp.name)

        # Tamper with file
        with open(tmp.name, "r") as f:
            payload = json.load(f)

        payload["dataset_hash"] = "tampered"

        with open(tmp.name, "w") as f:
            json.dump(payload, f)

        with pytest.raises(Exception):
            MetadataManager.load_metadata(tmp.name)

    os.remove(tmp.name)


# ======================================================
# DETERMINISTIC HASH
# ======================================================

def test_metadata_integrity_hash_deterministic():

    m1 = _build_valid_metadata()
    m2 = _build_valid_metadata()

    assert m1["metadata_integrity_hash"] == m2["metadata_integrity_hash"]