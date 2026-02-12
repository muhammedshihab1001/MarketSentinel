import os
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from core.time.market_time import MarketTime
from core.market.universe import MarketUniverse

from models.lstm_model import build_lstm_model


MODEL_DIR = "artifacts/lstm"

MODEL_PATH = f"{MODEL_DIR}/model.keras"
SCALER_PATH = f"{MODEL_DIR}/scalers.pkl"
METADATA_PATH = f"{MODEL_DIR}/metadata.json"

LOOKBACK_WINDOW = 60
EPOCHS = 50
MIN_ROWS_PER_TICKER = 900
MIN_SEQUENCES = 400

SEED = 42


############################################################
# STRICT DETERMINISM
############################################################

def configure_runtime():

    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # CPU-only ensures deterministic results
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(SEED)
    tf.random.set_seed(SEED)


############################################################
# FSYNC
############################################################

def _fsync_dir(directory):

    if os.name == "nt":
        return

    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


############################################################
# ATOMIC SAVE
############################################################

def atomic_save_model(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = path + ".tmp.keras"
    model.save(tmp_path)

    os.replace(tmp_path, path)
    _fsync_dir(os.path.dirname(path))


def atomic_joblib_dump(obj, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = path + ".tmp"
    joblib.dump(obj, tmp)

    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path))


############################################################
# LOAD DATA — CLOCK + UNIVERSE GOVERNED
############################################################

def load_data(start_date, end_date):

    fetcher = StockPriceFetcher()

    datasets = []
    universe = MarketUniverse.get_universe()

    print(f"Training universe size: {len(universe)}")

    for ticker in universe:

        df = fetcher.fetch(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or df.empty:
            continue

        if len(df) < MIN_ROWS_PER_TICKER:
            continue

        df["ticker"] = ticker
        datasets.append(df)

    # 🚨 Universe collapse protection
    if len(datasets) < 6:
        raise RuntimeError(
            "Universe collapse — too few assets survived."
        )

    df = pd.concat(datasets, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["ticker", "date"], inplace=True)

    return df.reset_index(drop=True)


############################################################
# SEQUENCE BUILDER
############################################################

def build_sequences_with_scalers(df):

    X_all = []
    y_all = []
    scalers = {}

    for ticker, tdf in df.groupby("ticker"):

        prices = tdf["close"].astype("float32").values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)

        scalers[ticker] = scaler

        for i in range(len(scaled) - LOOKBACK_WINDOW):
            X_all.append(scaled[i:i+LOOKBACK_WINDOW])
            y_all.append(scaled[i+LOOKBACK_WINDOW])

    if len(X_all) < MIN_SEQUENCES:
        raise RuntimeError("Insufficient sequences generated.")

    return (
        np.asarray(X_all, dtype=np.float32),
        np.asarray(y_all, dtype=np.float32),
        scalers
    )


def apply_scalers(df, scalers):

    X_all = []
    y_all = []

    for ticker, tdf in df.groupby("ticker"):

        if ticker not in scalers:
            continue

        prices = tdf["close"].astype("float32").values.reshape(-1, 1)
        scaled = scalers[ticker].transform(prices)

        for i in range(len(scaled) - LOOKBACK_WINDOW):
            X_all.append(scaled[i:i+LOOKBACK_WINDOW])
            y_all.append(scaled[i+LOOKBACK_WINDOW])

    if not X_all:
        raise RuntimeError("Validation produced zero sequences.")

    return (
        np.asarray(X_all, dtype=np.float32),
        np.asarray(y_all, dtype=np.float32)
    )


############################################################

def time_series_validation(df):

    split_date = df["date"].quantile(0.8)

    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    X_train, y_train, scalers = build_sequences_with_scalers(train_df)
    X_test, y_test = apply_scalers(test_df, scalers)

    model = build_lstm_model((LOOKBACK_WINDOW, 1))

    early = EarlyStopping(
        monitor="val_loss",
        patience=7,
        min_delta=1e-5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[early],
        verbose=1
    )

    val_loss = float(min(history.history["val_loss"]))

    if not np.isfinite(val_loss):
        raise RuntimeError("Training produced non-finite loss.")

    return model, scalers, val_loss


############################################################
# MAIN — INSTITUTIONAL CLOCK
############################################################

def main(start_date=None, end_date=None):

    configure_runtime()

    if start_date is None or end_date is None:
        start_date, end_date = MarketTime.training_window()

    print(f"Institutional LSTM Training | {start_date} -> {end_date}")

    df = load_data(start_date, end_date)

    dataset_hash = MetadataManager.fingerprint_dataset(
        df[["ticker", "date", "close"]]
    )

    model, scalers, val_loss = time_series_validation(df)

    atomic_save_model(model, MODEL_PATH)
    atomic_joblib_dump(scalers, SCALER_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="lstm_price_forecast",
        metrics={"val_loss": val_loss},
        features=["close_sequence"],
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="training_manifest_v1",
        extra_fields={
            "training_window": {
                "start": start_date,
                "end": end_date
            },
            "training_universe": MarketUniverse.get_universe()
        }
    )

    MetadataManager.save_metadata(metadata, METADATA_PATH)

    version = ModelRegistry.register_model(
        MODEL_DIR,
        MODEL_PATH,
        METADATA_PATH
    )

    scaler_target = os.path.join(MODEL_DIR, version, "scalers.pkl")

    os.replace(SCALER_PATH, scaler_target)

    _fsync_dir(os.path.join(MODEL_DIR, version))

    print(f"LSTM registered → {version}")


if __name__ == "__main__":
    main()
