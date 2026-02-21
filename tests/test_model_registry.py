import tempfile
import os
import json
import hashlib

from core.artifacts.model_registry import ModelRegistry
from core.artifacts.metadata_manager import MetadataManager
from core.schema.feature_schema import get_schema_signature


############################################################
# HELPER — create real binary artifact
############################################################

def create_binary_file(path, size_bytes=60000):
    with open(path, "wb") as f:
        f.write(os.urandom(size_bytes))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


############################################################
# MAIN TEST
############################################################

def test_model_registration_atomic_and_verified():

    with tempfile.TemporaryDirectory() as tmp:

        model_path = os.path.join(tmp, "model.pkl")
        meta_path = os.path.join(tmp, "metadata.json")

        # Create realistic artifact
        create_binary_file(model_path)

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
            registry_dir=tmp,
            model_artifact_path=model_path,
            metadata_path=meta_path
        )

        assert os.path.isdir(version_dir)

        ###################################################
        # MANIFEST VALIDATION
        ###################################################

        manifest_path = os.path.join(
            version_dir,
            ModelRegistry.MANIFEST_NAME
        )

        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["schema_signature"] == get_schema_signature()
        assert manifest["stage"] == "production"
        assert "artifacts" in manifest
        assert "model.pkl" in manifest["artifacts"]

        ###################################################
        # HASH VALIDATION
        ###################################################

        expected_hash = sha256(
            os.path.join(version_dir, "model.pkl")
        )

        assert manifest["artifacts"]["model.pkl"] == expected_hash

        ###################################################
        # VERIFY ARTIFACTS
        ###################################################

        version_name = os.path.basename(version_dir)

        # Should not raise
        ModelRegistry.verify_artifacts(tmp, version_name)