import time
import datetime
import os
import json
import logging

from training.train_xgboost import main as train_xgb


logger = logging.getLogger("marketsentinel.training")

RUNS_DIR = "artifacts/training_runs"

MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40


def save_manifest(run_id: str, manifest: dict):

    os.makedirs(RUNS_DIR, exist_ok=True)

    final_path = os.path.join(RUNS_DIR, f"{run_id}.json")
    temp_path = final_path + ".tmp"

    with open(temp_path, "w") as f:
        json.dump(manifest, f, indent=4)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, final_path)


def validate_metrics(metrics: dict):

    sharpe = metrics.get("avg_sharpe")
    drawdown = metrics.get("max_drawdown")

    if sharpe is None:
        raise RuntimeError("Training failed — missing sharpe.")

    if sharpe < MIN_SHARPE:
        raise RuntimeError("Model rejected — sharpe too low.")

    if drawdown is not None and drawdown < MAX_DRAWDOWN:
        raise RuntimeError("Model rejected — drawdown too severe.")


def main():

    run_id = datetime.datetime.utcnow().strftime(
        "run_%Y_%m_%d_%H%M%S"
    )

    start = time.time()

    try:

        logger.info("Starting XGBoost training...")

        metrics = train_xgb()

        validate_metrics(metrics)

        manifest = {
            "run_id": run_id,
            "status": "success",
            "metrics": metrics,
            "runtime_sec": round(time.time() - start, 2)
        }

        save_manifest(run_id, manifest)

        logger.info("Training completed successfully.")

    except Exception as e:

        logger.exception("Training failed.")

        manifest = {
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
            "runtime_sec": round(time.time() - start, 2)
        }

        save_manifest(run_id, manifest)

        raise


if __name__ == "__main__":
    main()
