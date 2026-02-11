import os
import datetime
import numpy as np
import pandas as pd

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from models.prophet_model import train_prophet

from prophet.serialize import model_to_json


MODEL_DIR = "artifacts/prophet"

TEMP_MODEL_PATH = f"{MODEL_DIR}/prophet_trend.json"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

TRAINING_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"
]

MIN_DATA_ROWS = 500
MAX_FORECAST_STD_RATIO = 0.25
MIN_FORECAST_STD = 0.0005

SEED = 42
EPSILON = 1e-6


# ---------------------------------------------------
# CPU STABILITY
# ---------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------

def validate_forecast(model):

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    preds = forecast["yhat"].tail(30).values

    std = float(np.std(preds))
    mean_val = float(np.mean(preds))

    slope = float(preds[-1] - preds[0])

    if std < MIN_FORECAST_STD:
        raise RuntimeError(
            "Prophet rejected: forecast collapsed"
        )

    ratio = std / max(abs(mean_val), EPSILON)

    if ratio > MAX_FORECAST_STD_RATIO:
        raise RuntimeError(
            "Prophet rejected: forecast too volatile"
        )

    return {
        "forecast_std": std,
        "trend_slope_30d": slope,
        "forecast_std_ratio": ratio
    }


# ---------------------------------------------------
# MULTI-ASSET LOADER
# ---------------------------------------------------

def load_training_data():

    fetcher = StockPriceFetcher()
    end_date = datetime.date.today().isoformat()

    datasets = []

    for ticker in TRAINING_TICKERS:

        df = fetcher.fetch(
            ticker=ticker,
            start_date="2018-01-01",
            end_date=end_date
        )

        if df.empty:
            continue

        df["ticker"] = ticker
        datasets.append(df)

    if not datasets:
        raise RuntimeError("No datasets fetched for Prophet.")

    df = pd.concat(datasets, ignore_index=True)

    if len(df) < MIN_DATA_ROWS:
        raise RuntimeError(
            f"Prophet rejected — insufficient rows ({len(df)})"
        )

    return df.sort_values(["ticker", "date"]), end_date


# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------

def train():

    np.random.seed(SEED)

    df, end_date = load_training_data()

    # Prophet expects ds/y
    prophet_df = (
        df.groupby("date")["close"]
        .mean()
        .reset_index()
    )

    prophet_df.rename(
        columns={"date": "date", "close": "close"},
        inplace=True
    )

    model = train_prophet(prophet_df)

    governance_metrics = validate_forecast(model)

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    metrics = {
        "training_rows": len(df),
        "asset_count": df["ticker"].nunique(),
        "last_price": float(df["close"].iloc[-1]),
        "data_span_days": int(
            (df["date"].max() - df["date"].min()).days
        ),
        **governance_metrics
    }

    metadata = MetadataManager.create_metadata(
        model_name="prophet_trend",
        metrics=metrics,
        features=["date", "close"],
        training_start="2018-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="model"
    )

    return model, metadata


# ---------------------------------------------------
# SAFE SERIALIZATION
# ---------------------------------------------------

def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w") as f:
        f.write(model_to_json(model))

    os.replace(tmp_path, path)


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

    version = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    print(f"Prophet registered → {version}")
