import os
import datetime
import uuid
from typing import Dict, Tuple

import numpy as np
import joblib

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry

from models.sarimax_model import SarimaxModel


MODEL_DIR = "artifacts/sarimax"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

TRAINING_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL",
    "META","TSLA","JPM","XOM","AVGO","AMD"
]

MIN_DATA_ROWS = 600
SEED = 42

np.random.seed(SEED)


########################################################
# DIRECTORY FSYNC
########################################################

def _fsync_dir(directory):

    if os.name == "nt":
        return

    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


########################################################
# ATOMIC SAVE
########################################################

def save_model_atomic(model: SarimaxModel, path: str) -> None:

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"

    joblib.dump(model, tmp_path)

    os.replace(tmp_path, path)

    _fsync_dir(directory)

    if not os.path.exists(path):
        raise RuntimeError("Model write failed.")


########################################################
# LOAD DATA
########################################################

def load_training_data() -> Tuple[Dict[str, object], str]:

    fetcher = StockPriceFetcher()

    end_date = datetime.date.today().isoformat()

    datasets = {}

    for ticker in TRAINING_TICKERS:

        df = fetcher.fetch(
            ticker=ticker,
            start_date="2014-01-01",
            end_date=end_date
        )

        if df is None or df.empty or len(df) < MIN_DATA_ROWS:
            continue

        datasets[ticker] = df[["date", "close"]].copy()

    if not datasets:
        raise RuntimeError("No datasets available for SARIMAX.")

    return datasets, end_date


########################################################
# RISK-ADJUSTED SCORING
########################################################

def score_model(metrics):

    slope = abs(metrics["normalized_slope"])
    vol = max(metrics["forecast_volatility"], 1e-6)

    # Sharpe-lite
    return slope / vol


########################################################
# TRAIN CHAMPION
########################################################

def train_champion():

    datasets, end_date = load_training_data()

    best_model = None
    best_score = -np.inf
    best_ticker = None
    best_metrics = None

    rejection_log = []

    for ticker, df in datasets.items():

        try:

            model = SarimaxModel()
            model.fit(df)

            metrics = model.forecast()

            score = score_model(metrics)

            print(f"SARIMAX {ticker} score={round(score,4)}")

            if score > best_score:
                best_model = model
                best_score = score
                best_ticker = ticker
                best_metrics = metrics

        except Exception as exc:
            msg = f"{ticker} rejected: {str(exc)}"
            rejection_log.append(msg)
            print(msg)

    if best_model is None:
        raise RuntimeError(
            "All SARIMAX models rejected.\n"
            + "\n".join(rejection_log)
        )

    dataset_hash = MetadataManager.fingerprint_dataset(
        datasets[best_ticker]
    )

    training_range = best_model.get_training_range()

    metadata = MetadataManager.create_metadata(
        model_name="sarimax_trend",
        metrics={
            "champion_asset": best_ticker,
            "risk_adjusted_score": float(best_score),
            **best_metrics
        },
        features=["date", "close"],
        training_start=training_range["start"],
        training_end=training_range["end"],
        dataset_hash=dataset_hash,
        metadata_type="sequence",
        extra_fields={
            "model_type": "SARIMAX",
            "parameters": best_model.get_params()
        }
    )

    return best_model, metadata


########################################################
# MAIN
########################################################

if __name__ == "__main__":

    print("Institutional SARIMAX Training")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model, metadata = train_champion()

    save_model_atomic(model, TEMP_MODEL_PATH)

    MetadataManager.save_metadata(
        metadata,
        TEMP_METADATA_PATH
    )

    version = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    # CRITICAL — verify before promotion
    ModelRegistry.verify_artifacts(
        MODEL_DIR,
        version
    )

    # promote pointer
    ModelRegistry.promote_to_latest(
        MODEL_DIR,
        version
    )

    print(f"SARIMAX champion registered -> {version}")
