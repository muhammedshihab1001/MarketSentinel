import os
import datetime
import joblib

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from models.prophet_model import train_prophet


MODEL_DIR = "artifacts/prophet"
MODEL_PATH = f"{MODEL_DIR}/prophet_trend.pkl"
METADATA_PATH = f"{MODEL_DIR}/metadata.json"


def train():

    end_date = datetime.date.today().isoformat()

    fetcher = StockPriceFetcher()

    df = fetcher.fetch(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date=end_date
    )

    model = train_prophet(df)

    return model, end_date


if __name__ == "__main__":

    model, end_date = train()

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="prophet_trend",
        metrics={"status": "trained"},
        features=["date", "close"],
        training_start="2018-01-01",
        training_end=end_date
    )

    MetadataManager.save_metadata(metadata, METADATA_PATH)

    print("Prophet model saved")
    print("Metadata saved")
