import json
import pytest

from core.artifacts.metadata_manager import MetadataManager
from core.schema.feature_schema import MODEL_FEATURES


############################################################
# VALID METADATA BUILDER
############################################################

def _build_valid_metadata():
    """Build valid metadata using create_metadata().

    Note: artifact_hash, schema_signature, schema_version, etc. are
    set internally by create_metadata() — they are IMMUTABLE and
    cannot be passed via extra_fields.
    """

    return MetadataManager.create_metadata(
        model_name="xgboost",
        metrics={"sharpe": 1.2},
        features=MODEL_FEATURES,
        training_start="2025-01-01",
        training_end="2026-01-01",
        dataset_hash="abc" * 22,  # needs to look like a real hash
        dataset_rows=1000,
        metadata_type="training_manifest_v1",
        feature_checksum="dummy_checksum",
        extra_fields={
            "model_version": "test_v1",
        }
    )


############################################################
# BASIC INTEGRITY ROUNDTRIP
############################################################

def test_metadata_integrity_roundtrip(tmp_path):

    metadata = _build_valid_metadata()

    path = str(tmp_path / "metadata.json")

    MetadataManager.save_metadata(metadata, path)

    loaded = MetadataManager.load_metadata(path)

    assert loaded["metadata_integrity_hash"] == metadata["metadata_integrity_hash"]
    assert loaded["dataset_hash"] == metadata["dataset_hash"]


############################################################
# REQUIRED FIELDS PRESENT
############################################################

def test_metadata_required_fields_present():

    metadata = _build_valid_metadata()

    required_keys = {
        "metadata_integrity_hash",
        "metadata_type",
        "dataset_hash",
        "dataset_rows",
        "features",
        "training_window",
    }

    assert required_keys.issubset(metadata.keys())

    # training_window contains start/end
    assert "start" in metadata["training_window"]
    assert "end" in metadata["training_window"]


############################################################
# JSON SERIALIZATION SAFETY
############################################################

def test_metadata_is_json_serializable():

    metadata = _build_valid_metadata()

    payload = json.dumps(metadata)

    assert isinstance(payload, str)


############################################################
# TAMPER DETECTION
############################################################

def test_metadata_tamper_detection(tmp_path):

    metadata = _build_valid_metadata()

    path = str(tmp_path / "metadata.json")

    MetadataManager.save_metadata(metadata, path)

    with open(path, "r") as f:
        payload = json.load(f)

    payload["dataset_hash"] = "tampered"

    with open(path, "w") as f:
        json.dump(payload, f)

    with pytest.raises(Exception):
        MetadataManager.load_metadata(path)


############################################################
# DETERMINISTIC HASH
############################################################

def test_metadata_integrity_hash_deterministic():

    m1 = _build_valid_metadata()
    m2 = _build_valid_metadata()

    assert m1["metadata_integrity_hash"] == m2["metadata_integrity_hash"]


############################################################
# HASH EXISTS AND LOOKS VALID
############################################################

def test_metadata_hash_format():

    metadata = _build_valid_metadata()

    hash_value = metadata["metadata_integrity_hash"]

    assert isinstance(hash_value, str)
    assert len(hash_value) >= 32
