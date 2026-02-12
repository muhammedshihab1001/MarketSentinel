import os
import uuid
from typing import Dict, Tuple

import numpy as np
import joblib
import pandas as pd

from statsmodels.tsa.stattools import adfuller

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse

from models.sarimax_model import SarimaxModel


MODEL_DIR = "artifacts/sarimax"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

MIN_DATA_ROWS = 700
MIN_UNIVERSE = 5
VOL_FLOOR = 1e-4
VAR_FLOOR = 1e-8

SEED = 42


########################################################
# STRICT DETERMINISM
########################################################

def enforce_determinism():

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    np.random.seed(SEED)


########################################################
# FSYNC
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

def save_model_atomic(model: SarimaxModel, path: str):

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"

    joblib.dump(model, tmp_path)

    os.replace(tmp_path, path)
    _fsync_dir(directory)

    if not os.path.exists(path):
        raise RuntimeError("Model write failed.")


########################################################
# HARD STATIONARITY CHECK
########################################################

def assert_stationary(series: pd.Series):

    series = series.dropna()

    if len(series) < 200:
        raise RuntimeError("Series too short for stationarity test.")

    if series.var() < VAR_FLOOR:
        raise RuntimeError("Variance collapsed — unusable series.")

    result = adfuller(series)

    p_value = result[1]

    if not np.isfinite(p_value):
        raise RuntimeError("ADF produced invalid p-value.")

    if p_value > 0.05:
        raise RuntimeError(
            f"Non-stationary series detected (p={p_value})."
        )


########################################################
# LOAD DATA — GOVERNED
########################################################

def load_training_data(
    start_date: str,
    end_date: str
) -> Tuple[Dict[str, pd.DataFrame], list]:

    fetcher = StockPriceFetcher()
    universe = MarketUniverse.get_universe()

    print(f"SARIMAX universe size: {len(universe)}")

    datasets = {}
    surviving = []

    for ticker in universe:

        df = fetcher.fetch(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or len(df) < MIN_DATA_ROWS:
            continue

        df = df[["date", "close"]].copy()

        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        if df["close"].isna().any():
            continue

        if (df["close"] <= 0).any():
            continue

        log_returns = np.log(df["close"]).diff()

        assert_stationary(log_returns)

        datasets[ticker] = df
        surviving.append(ticker)

    if len(datasets) < MIN_UNIVERSE:
        raise RuntimeError(
            "Universe collapse — too few assets survived."
        )

    return datasets, surviving


########################################################
# RISK-ADJUSTED SCORING
########################################################

def score_model(metrics):

    slope = float(metrics["normalized_slope"])
    vol = max(float(metrics["forecast_volatility"]), VOL_FLOOR)

    if not np.isfinite(slope) or not np.isfinite(vol):
        raise RuntimeError("Invalid forecast metrics.")

    return abs(slope) / vol


########################################################
# TRAIN CHAMPION
########################################################

def train_champion(start_date, end_date):

    datasets, surviving = load_training_data(start_date, end_date)

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

    metadata = MetadataManager.create_metadata(
        model_name="sarimax_trend",
        metrics={
            "champion_asset": best_ticker,
            "risk_adjusted_score": float(best_score),
            **best_metrics
        },
        features=["close"],
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="timeseries_manifest_v1",
        extra_fields={
            "model_type": "SARIMAX",
            "parameters": best_model.get_params(),
            "training_universe": surviving,
            "universe_hash": MetadataManager.hash_list(surviving)
        }
    )

    return best_model, metadata


########################################################
# MAIN — MODEL GOVERNED CLOCK
########################################################

def main(start_date=None, end_date=None):

    enforce_determinism()

    if start_date is None or end_date is None:
        start_date, end_date = MarketTime.window_for("sarimax")

    print(
        f"Institutional SARIMAX Training | {start_date} -> {end_date}"
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    model, metadata = train_champion(start_date, end_date)

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

    ModelRegistry.verify_artifacts(
        MODEL_DIR,
        version
    )

    print(
        f"SARIMAX candidate registered -> {version}\n"
        "Promotion requires governance approval."
    )


if __name__ == "__main__":
    main()
