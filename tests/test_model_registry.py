import tempfile
import os
import json

from core.artifacts.model_registry import ModelRegistry
from core.artifacts.metadata_manager import MetadataManager


def test_model_registration_atomic():

    with tempfile.TemporaryDirectory() as tmp:

        model_path = os.path.join(tmp, "model.pkl")
        meta_path = os.path.join(tmp, "metadata.json")

        # Create fake model artifact
        with open(model_path, "w") as f:
            f.write("test-model")

        # Create VALID metadata
        metadata = MetadataManager.create_metadata(
            model_name="test_model",
            metrics={"accuracy": 0.9},
            features=["f1", "f2"],
            training_start="2020-01-01",
            training_end="2021-01-01",
            dataset_hash="abc123",
            metadata_type="model"
        )

        MetadataManager.save_metadata(metadata, meta_path)

        version_dir = ModelRegistry.register_model(
            tmp,
            model_path,
            meta_path
        )

        assert os.path.exists(version_dir)

        # Ensure manifest created
        manifest = os.path.join(version_dir, "manifest.json")
        assert os.path.exists(manifest)
