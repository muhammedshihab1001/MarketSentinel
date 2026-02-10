import os
import datetime
import joblib

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from models.prophet_model import train_prophet


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

MODEL_DIR = "artifacts/prophet"

TEMP_MODEL_PATH = f"{MODEL_DIR}/prophet_trend.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"


# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------

def train():

    print("\n🚀 Starting Prophet training...\n")

    end_date = datetime.date.today().isoformat()

    fetcher = StockPriceFetcher()

    df = fetcher.fetch(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date=end_date
    )

    if df.empty:
        raise ValueError("No price data fetched for Prophet training")

    model = train_prophet(df)

    # ---------------------------------------------------
    # DATASET FINGERPRINT (CRITICAL)
    # ---------------------------------------------------

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    metadata = MetadataManager.create_metadata(
        model_name="prophet_trend",
        metrics={"status": "trained"},
        features=["date", "close"],
        training_start="2018-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash
    )

    return model, metadata


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":

    os.makedirs(MODEL_DIR, exist_ok=True)

    model, metadata = train()

    # Save artifacts temporarily
    joblib.dump(model, TEMP_MODEL_PATH)
    MetadataManager.save_metadata(metadata, TEMP_METADATA_PATH)

    # ---------------------------------------------------
    # REGISTER MODEL (VERY IMPORTANT)
    # ---------------------------------------------------

    version_dir = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print("Prophet model registered successfully.")
    print(f"Version directory: {version_dir}")
