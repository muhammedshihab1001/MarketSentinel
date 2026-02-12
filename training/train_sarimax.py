import os
import datetime
import tempfile
import shutil
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
# ATOMIC SAVE
########################################################

def save_model_atomic(model: SarimaxModel, path: str) -> None:

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=os.path.dirname(path)
    ) as tmp:

        joblib.dump(model, tmp.name)
        temp_name = tmp.name

    shutil.move(temp_name, path)

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

            score = abs(metrics["normalized_slope"])

            if score > best_score:
                best_model = model
                best_score = score
                best_ticker = ticker
                best_metrics = metrics

        except Exception as exc:
            rejection_log.append(f"{ticker}: {str(exc)}")

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
            "score": float(best_score),
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

    print(f"SARIMAX champion registered -> {version}")
