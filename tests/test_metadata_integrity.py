import json
import tempfile
from core.artifacts.metadata_manager import MetadataManager


def test_metadata_integrity():

    metadata = MetadataManager.create_metadata(
        model_name="xgboost",
        metrics={"sharpe": 1.2},
        features=tuple(["f1"] * 10),
        training_start="2025-01-01",
        training_end="2026-01-01",
        dataset_hash="abc",
        dataset_rows=1000,
        metadata_type="training_manifest_v1"
    )

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        MetadataManager.save_metadata(metadata, tmp.name)
        loaded = MetadataManager.load_metadata(tmp.name)

    assert loaded["metadata_integrity_hash"] == metadata["metadata_integrity_hash"]