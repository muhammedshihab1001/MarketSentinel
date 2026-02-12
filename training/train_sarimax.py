import os
import datetime
import numpy as np

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry

from models.sarimax_model import (
    train_sarimax,
    forecast_sarimax
)


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

def save_model_atomic(model, path):

    import joblib
    import tempfile
    import shutil

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

def load_training_data():

    fetcher = StockPriceFetcher()

    end_date = datetime.date.today().isoformat()

    datasets = {}

    for ticker in TRAINING_TICKERS:

        df = fetcher.fetch(
            ticker=ticker,
            start_date="2014-01-01",
            end_date=end_date
        )

        if df.empty or len(df) < MIN_DATA_ROWS:
            continue

        datasets[ticker] = df[["date","close"]]

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

    for ticker, df in datasets.items():

        try:

            model = train_sarimax(df)

            metrics = forecast_sarimax(model)

            score = abs(metrics["normalized_slope"])

            if score > best_score:
                best_model = model
                best_score = score
                best_ticker = ticker
                best_metrics = metrics

        except Exception as e:

            print(f"SARIMAX rejected for {ticker}: {e}")
            continue

    if best_model is None:
        raise RuntimeError("All SARIMAX models rejected.")

    dataset_hash = MetadataManager.fingerprint_dataset(
        datasets[best_ticker]
    )

    metadata = MetadataManager.create_metadata(
        model_name="sarimax_trend",
        metrics={
            "champion_asset": best_ticker,
            "score": float(best_score),
            **best_metrics
        },
        features=["date","close"],
        training_start="2014-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="sequence"
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

    print(f"SARIMAX champion registered → {version}")
