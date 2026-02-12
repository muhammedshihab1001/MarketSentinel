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
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
    "JPM","XOM","AVGO","AMD"
]

MIN_DATA_ROWS = 700
MAX_FORECAST_STD_RATIO = 0.30
MIN_FORECAST_STD = 0.0007
MAX_TREND_JUMP = 0.25

SEED = 42
EPSILON = 1e-6


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def validate_forecast(model):

    future = model.make_future_dataframe(periods=45)
    forecast = model.predict(future)

    preds = forecast["yhat"].tail(45).values

    if np.isnan(preds).any():
        raise RuntimeError("Prophet rejected: NaN forecast")

    std = float(np.std(preds))
    mean_val = float(np.mean(preds))
    slope = float(preds[-1] - preds[0])

    if std < MIN_FORECAST_STD:
        raise RuntimeError("Prophet rejected: flat forecast")

    ratio = std / max(abs(mean_val), EPSILON)

    if ratio > MAX_FORECAST_STD_RATIO:
        raise RuntimeError("Prophet rejected: volatility explosion")

    trend_jump = abs(slope) / max(abs(mean_val), EPSILON)

    if trend_jump > MAX_TREND_JUMP:
        raise RuntimeError("Prophet rejected: unrealistic trend")

    return {
        "forecast_std": std,
        "trend_slope_45d": slope,
        "forecast_std_ratio": ratio,
        "trend_jump_ratio": trend_jump
    }


def load_training_data():

    fetcher = StockPriceFetcher()
    end_date = datetime.date.today().isoformat()

    datasets = {}

    for ticker in TRAINING_TICKERS:

        df = fetcher.fetch(
            ticker=ticker,
            start_date="2015-01-01",
            end_date=end_date
        )

        if df.empty or len(df) < MIN_DATA_ROWS:
            continue

        df = df.sort_values("date")

        prophet_df = df[["date", "close"]].rename(
            columns={"date": "ds", "close": "y"}
        )

        datasets[ticker] = prophet_df

    if not datasets:
        raise RuntimeError("No valid datasets for Prophet training.")

    return datasets, end_date


def train_champion():

    np.random.seed(SEED)

    datasets, end_date = load_training_data()

    best_model = None
    best_score = -np.inf
    best_ticker = None
    best_metrics = None

    for ticker, df in datasets.items():

        try:

            model = train_prophet(df)

            metrics = validate_forecast(model)

            score = (
                abs(metrics["trend_slope_45d"]) /
                max(metrics["forecast_std"], EPSILON)
            )

            if score > best_score:
                best_model = model
                best_score = score
                best_ticker = ticker
                best_metrics = metrics

        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("All Prophet models rejected.")

    dataset_hash = MetadataManager.fingerprint_dataset(
        datasets[best_ticker]
    )

    metadata = MetadataManager.create_metadata(
        model_name="prophet_trend",
        metrics={
            "champion_asset": best_ticker,
            "score": float(best_score),
            **best_metrics
        },
        features=["ds", "y"],
        training_start="2015-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="model"
    )

    return best_model, metadata


def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w") as f:
        f.write(model_to_json(model))

    os.replace(tmp_path, path)


if __name__ == "__main__":

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

    print(f"Prophet champion registered → {version}")
