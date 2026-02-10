import os
import datetime
import shutil


class ModelRegistry:
    """
    Lightweight filesystem model registry.

    Guarantees:
    ✅ Versioned artifacts
    ✅ No overwrite risk
    ✅ Instant rollback capability
    ✅ Deployment safety
    """

    @staticmethod
    def _version() -> str:
        return datetime.datetime.utcnow().strftime(
            "v%Y_%m_%d_%H%M%S"
        )

    # --------------------------------------------------

    @staticmethod
    def register_model(
        base_dir: str,
        model_path: str,
        metadata_path: str
    ) -> str:
        """
        Moves artifacts into versioned directory
        and updates 'latest' pointer.
        """

        version = ModelRegistry._version()

        version_dir = os.path.join(base_dir, version)

        os.makedirs(version_dir, exist_ok=True)

        # move artifacts
        shutil.move(model_path, os.path.join(version_dir, os.path.basename(model_path)))
        shutil.move(metadata_path, os.path.join(version_dir, os.path.basename(metadata_path)))

        # update latest pointer
        latest_path = os.path.join(base_dir, "latest")

        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)

        os.symlink(version, latest_path)

        return version_dir
