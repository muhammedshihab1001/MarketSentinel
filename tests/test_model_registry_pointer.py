import tempfile
import os
import json
import hashlib

from core.artifacts.model_registry import ModelRegistry
from core.artifacts.metadata_manager import MetadataManager
from core.schema.feature_schema import get_schema_signature


############################################################
# HELPER
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
# POINTER INTEGRITY TEST
############################################################

def test_latest_pointer_integrity():

    with tempfile.TemporaryDirectory() as d:

        model_path = os.path.join(d, "model.pkl")
        meta_path = os.path.join(d, "metadata.json")

        # Real artifact
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

        ###################################################
        # REGISTER MODEL
        ###################################################

        version_dir = ModelRegistry.register_model(
            registry_dir=d,
            model_artifact_path=model_path,
            metadata_path=meta_path
        )

        version = os.path.basename(version_dir)

        ###################################################
        # LATEST POINTER RESOLUTION
        ###################################################

        latest = ModelRegistry.get_latest_version(d)

        assert latest == version

        ###################################################
        # VERIFY THROUGH POINTER
        ###################################################

        ModelRegistry.verify_artifacts(d, latest)

        ###################################################
        # MANIFEST CHECK
        ###################################################

        manifest_path = os.path.join(
            d,
            latest,
            ModelRegistry.MANIFEST_NAME
        )

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["schema_signature"] == get_schema_signature()
        assert manifest["stage"] == "production"
        assert "model.pkl" in manifest["artifacts"]

        ###################################################
        # HASH MATCH CHECK
        ###################################################

        expected_hash = sha256(
            os.path.join(d, latest, "model.pkl")
        )

        assert manifest["artifacts"]["model.pkl"] == expected_hash