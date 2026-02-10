import os
import datetime
import tempfile
import shutil

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from models.prophet_model import train_prophet

from prophet.serialize import model_to_json


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

MODEL_DIR = "artifacts/prophet"

TEMP_MODEL_PATH = f"{MODEL_DIR}/prophet_trend.json"
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
    # DATASET FINGERPRINT
    # ---------------------------------------------------

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    # 🔥 Institutional metrics
    metrics = {
        "training_rows": len(df),
        "last_price": float(df["close"].iloc[-1]),
        "data_span_days": (df["date"].max() - df["date"].min()).days
    }

    metadata = MetadataManager.create_metadata(
        model_name="prophet_trend",
        metrics=metrics,
        features=["date", "close"],
        training_start="2018-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash
    )

    return model, metadata


# ---------------------------------------------------
# SAFE SERIALIZATION
# ---------------------------------------------------

def save_model_atomic(model, path):
    """
    Writes Prophet safely.

    Prevents corrupted artifacts if training crashes.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=os.path.dirname(path)
    ) as tmp:

        tmp.write(model_to_json(model))
        temp_name = tmp.name

    shutil.move(temp_name, path)


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":

    os.makedirs(MODEL_DIR, exist_ok=True)

    model, metadata = train()

    # Safe write
    save_model_atomic(model, TEMP_MODEL_PATH)

    MetadataManager.save_metadata(
        metadata,
        TEMP_METADATA_PATH
    )

    # ---------------------------------------------------
    # REGISTER MODEL
    # ---------------------------------------------------

    version_dir = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print("✅ Prophet model registered successfully.")
    print(f"📦 Version directory: {version_dir}")
