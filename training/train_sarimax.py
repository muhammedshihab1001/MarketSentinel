import os
import uuid
import logging
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


logger = logging.getLogger("marketsentinel.sarimax")

MODEL_DIR = "artifacts/sarimax"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

MIN_DATA_ROWS = 700
MIN_UNIVERSE = 5
VOL_FLOOR = 1e-4
VAR_FLOOR = 1e-8
MAX_MISSING_RATIO = 0.01

SEED = 42


########################################################
# STRICT DETERMINISM
########################################################

def enforce_determinism():

    os.environ["PYTHONHASHSEED"] = str(SEED)
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

    if len(series) < 250:
        raise RuntimeError("Series too short for stationarity test.")

    if not np.isfinite(series).all():
        raise RuntimeError("Non-finite values in returns.")

    if series.var() < VAR_FLOOR:
        raise RuntimeError("Variance collapsed — unusable series.")

    result = adfuller(series, autolag="AIC")

    p_value = result[1]

    if not np.isfinite(p_value):
        raise RuntimeError("ADF produced invalid p-value.")

    if p_value > 0.05:
        raise RuntimeError(
            f"Non-stationary series detected (p={p_value})."
        )


########################################################
# DATA CLEANING (NEW — VERY IMPORTANT)
########################################################

def clean_price_frame(df: pd.DataFrame):

    if df is None or df.empty:
        return None

    df = df[["date", "close"]].copy()

    df["date"] = pd.to_datetime(df["date"], utc=True)

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    if df["close"].isna().mean() > MAX_MISSING_RATIO:
        return None

    df = df.dropna()

    if (df["close"] <= 0).any():
        return None

    if not df["date"].is_monotonic_increasing:
        df = df.sort_values("date")

    df = df.drop_duplicates("date")

    if len(df) < MIN_DATA_ROWS:
        return None

    return df.reset_index(drop=True)


########################################################
# LOAD DATA
########################################################

def load_training_data(
    start_date: str,
    end_date: str
) -> Tuple[Dict[str, pd.DataFrame], list]:

    fetcher = StockPriceFetcher()
    universe = MarketUniverse.get_universe()

    logger.info("SARIMAX universe size=%s", len(universe))

    datasets = {}
    surviving = []

    for ticker in sorted(universe):  # deterministic

        try:

            df = fetcher.fetch(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )

            df = clean_price_frame(df)

            if df is None:
                continue

            log_returns = np.log(df["close"]).diff()

            assert_stationary(log_returns)

            datasets[ticker] = df
            surviving.append(ticker)

        except Exception as exc:

            logger.warning(
                "Ticker rejected | %s | %s",
                ticker,
                str(exc)
            )

    if len(datasets) < MIN_UNIVERSE:
        raise RuntimeError(
            "Universe collapse — too few assets survived."
        )

    logger.info(
        "Survivors=%s/%s",
        len(datasets),
        len(universe)
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

    rejection_log = {}

    for ticker in sorted(datasets.keys()):

        df = datasets[ticker]

        try:

            model = SarimaxModel()
            model.fit(df)

            metrics = model.forecast()

            score = score_model(metrics)

            logger.info(
                "SARIMAX %s score=%.4f",
                ticker,
                score
            )

            if (
                score > best_score
                or (score == best_score and ticker < best_ticker)
            ):
                best_model = model
                best_score = score
                best_ticker = ticker
                best_metrics = metrics

        except Exception as exc:

            rejection_log[ticker] = str(exc)

    if best_model is None:
        raise RuntimeError("All SARIMAX models rejected.")

    ####################################################
    # DATASET HASH (CHAMPION ONLY — lineage safe)
    ####################################################

    dataset_hash = MetadataManager.fingerprint_dataset(
        datasets[best_ticker]
    )

    metadata = MetadataManager.create_metadata(
        model_name="sarimax_trend",
        metrics={
            "champion_asset": best_ticker,
            "risk_adjusted_score": float(best_score),
            "model_fingerprint": best_model.fingerprint(),
            **best_metrics
        },
        features=["close"],
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        dataset_rows=len(datasets[best_ticker]),
        metadata_type="timeseries_manifest_v1",
        extra_fields={

            "model_type": "SARIMAX",
            "parameters": best_model.get_params(),

            "surviving_universe": sorted(surviving),
            "survivor_count": len(surviving),

            "rejections": rejection_log
        }
    )

    return best_model, metadata


########################################################
# MAIN
########################################################

def main(start_date=None, end_date=None):

    enforce_determinism()

    if start_date is None or end_date is None:
        start_date, end_date = MarketTime.window_for("sarimax")

    logger.info(
        "Institutional SARIMAX Training | %s -> %s",
        start_date,
        end_date
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

    logger.info(
        "SARIMAX candidate registered -> %s | promotion requires approval.",
        version
    )


if __name__ == "__main__":
    main()
