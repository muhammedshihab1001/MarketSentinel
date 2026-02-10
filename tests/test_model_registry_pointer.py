import tempfile
import json
import os
from core.artifacts.model_registry import ModelRegistry


def test_latest_pointer_created():

    with tempfile.TemporaryDirectory() as d:

        model = os.path.join(d, "model.pkl")
        meta = os.path.join(d, "metadata.json")

        open(model, "w").write("x")

        json.dump({
            "model_name": "test",
            "features": [],
            "metrics": {},
            "dataset_hash": "abc",
            "schema_signature": "sig"
        }, open(meta, "w"))

        version_dir = ModelRegistry.register_model(d, model, meta)

        version = os.path.basename(version_dir)

        ModelRegistry._write_latest_pointer(d, version)

        latest = ModelRegistry.get_latest_version(d)

        assert latest == version
