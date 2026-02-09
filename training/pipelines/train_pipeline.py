"""
Master Training Pipeline

Controls the full training workflow:
- XGBoost
- LSTM
- Prophet
"""

import subprocess
import sys


def run_script(module_path: str):
    """
    Runs a module as a subprocess.
    Safer than importing training scripts directly.
    Prevents memory collisions (important for TF).
    """

    print(f"\n Running {module_path}...\n")

    result = subprocess.run(
        [sys.executable, "-m", module_path],
        check=True
    )

    if result.returncode == 0:
        print(f" Completed {module_path}")
    else:
        raise RuntimeError(f" Failed {module_path}")


def main():

    pipeline_steps = [
        "training.train_xgboost",
        "training.train_lstm",
        "training.train_prophet",
    ]

    for step in pipeline_steps:
        run_script(step)

    print("\n FULL TRAINING PIPELINE COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    main()
