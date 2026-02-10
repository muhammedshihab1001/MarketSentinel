import os
import datetime
import tempfile
import shutil
import numpy as np

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from models.prophet_model import train_prophet

from prophet.serialize import model_to_json


MODEL_DIR = "artifacts/prophet"

TEMP_MODEL_PATH = f"{MODEL_DIR}/prophet_trend.json"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"


MIN_DATA_ROWS = 500
MAX_FORECAST_STD = 0.25
MIN_FORECAST_STD = 0.0005


# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------

def validate_forecast(model):

    future = model.make_future_dataframe(periods=30)

    forecast = model.predict(future)

    preds = forecast["yhat"].tail(30).values

    std = float(np.std(preds))

    slope = float(preds[-1] - preds[0])

    if std < MIN_FORECAST_STD:
        raise RuntimeError("Prophet rejected: forecast collapsed (near zero variance)")

    if std > MAX_FORECAST_STD * abs(preds.mean()):
        raise RuntimeError("Prophet rejected: forecast too volatile")

    return {
        "forecast_std": std,
        "trend_slope_30d": slope
    }


# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------

def train():

    end_date = datetime.date.today().isoformat()

    fetcher = StockPriceFetcher()

    df = fetcher.fetch(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date=end_date
    )

    if df.empty:
        raise ValueError("No price data fetched for Prophet training")

    if len(df) < MIN_DATA_ROWS:
        raise RuntimeError("Prophet rejected: insufficient training history")

    model = train_prophet(df)

    governance_metrics = validate_forecast(model)

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    metrics = {
        "training_rows": len(df),
        "last_price": float(df["close"].iloc[-1]),
        "data_span_days": (df["date"].max() - df["date"].min()).days,
        **governance_metrics
    }

    metadata = MetadataManager.create_metadata(
        model_name="prophet_trend",
        metrics=metrics,
        features=["date", "close"],
        training_start="2018-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash
    )

    metadata["schema_version"] = "3.0"

    return model, metadata


# ---------------------------------------------------
# SAFE SERIALIZATION
# ---------------------------------------------------

def save_model_atomic(model, path):

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

    save_model_atomic(model, TEMP_MODEL_PATH)

    MetadataManager.save_metadata(
        metadata,
        TEMP_METADATA_PATH
    )

    version_dir = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print("Prophet model registered successfully.")
    print(f"Version directory: {version_dir}")
