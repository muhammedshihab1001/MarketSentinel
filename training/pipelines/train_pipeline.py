"""
MarketSentinel Master Training Pipeline

Single entrypoint for ALL model training.

This acts as the training control plane.
"""

import subprocess
import sys
import time


PIPELINE_STEPS = [
    ("XGBoost", "training.train_xgboost"),
    ("LSTM", "training.train_lstm"),
    ("Prophet", "training.train_prophet"),
]


def run_step(name: str, module: str):

    print("\n===================================")
    print(f" Starting {name} training")
    print("===================================\n")

    start = time.time()

    result = subprocess.run(
        [sys.executable, "-m", module],
        check=True
    )

    duration = round(time.time() - start, 2)

    if result.returncode == 0:
        print(f"\n {name} completed in {duration}s\n")
    else:
        raise RuntimeError(f"{name} training failed")


def main():

    total_start = time.time()

    print("\n########################################")
    print("   MARKET SENTINEL — TRAINING RUN")
    print("########################################\n")

    for name, module in PIPELINE_STEPS:
        run_step(name, module)

    total_time = round(time.time() - total_start, 2)

    print("\n########################################")
    print(f" ALL MODELS TRAINED SUCCESSFULLY")
    print(f" Total runtime: {total_time}s")
    print("########################################\n")


if __name__ == "__main__":
    main()
