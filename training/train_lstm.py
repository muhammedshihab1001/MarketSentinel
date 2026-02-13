import os
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

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
EPOCHS = 60
MIN_ROWS_PER_TICKER = 950
MIN_SEQUENCES = 600

SEED = 42


############################################################
# STRICT DETERMINISM
############################################################

def configure_runtime():

    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    tf.config.experimental.enable_op_determinism()


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
# LOAD DATA — NO SURVIVORSHIP BIAS
############################################################

def load_data(start_date, end_date):

    fetcher = StockPriceFetcher()
    universe = MarketUniverse.get_universe()

    datasets = []
    surviving = []

    for ticker in universe:

        df = fetcher.fetch(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or len(df) < MIN_ROWS_PER_TICKER:
            continue

        if (df["close"] <= 0).any():
            continue

        df["ticker"] = ticker
        surviving.append(ticker)
        datasets.append(df)

    if len(datasets) < 6:
        raise RuntimeError("Universe collapse — too few assets survived.")

    df = pd.concat(datasets, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.sort_values(["ticker", "date"], inplace=True)

    return df.reset_index(drop=True), surviving


############################################################
# RETURN MODELING (CRITICAL UPGRADE)
############################################################

def compute_returns(df):

    df = df.copy()

    df["log_price"] = np.log(df["close"].astype("float64"))
    df["return"] = df.groupby("ticker")["log_price"].diff()

    df.dropna(inplace=True)

    if not np.isfinite(df["return"]).all():
        raise RuntimeError("Non-finite returns detected.")

    return df


############################################################
# SEQUENCE BUILDER — VOL NORMALIZED
############################################################

def build_sequences(df):

    X_all = []
    y_all = []
    scalers = {}

    for ticker, tdf in df.groupby("ticker"):

        returns = tdf["return"].values.reshape(-1, 1)

        if np.std(returns) < 1e-6:
            continue

        scaler = RobustScaler()
        scaled = scaler.fit_transform(returns)

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

        returns = tdf["return"].values.reshape(-1, 1)
        scaled = scalers[ticker].transform(returns)

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
# TIME SAFE VALIDATION — PER TICKER SPLIT
############################################################

def time_series_validation(df):

    train_parts = []
    test_parts = []

    for ticker, tdf in df.groupby("ticker"):

        split_idx = int(len(tdf) * 0.8)

        train_parts.append(tdf.iloc[:split_idx])
        test_parts.append(tdf.iloc[split_idx:])

    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)

    X_train, y_train, scalers = build_sequences(train_df)
    X_test, y_test = apply_scalers(test_df, scalers)

    model = build_lstm_model((LOOKBACK_WINDOW, 1))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=1e-5,
            restore_best_weights=True
        ),
        TerminateOnNaN()
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    val_loss = float(min(history.history["val_loss"]))

    if not np.isfinite(val_loss):
        raise RuntimeError("Training produced non-finite loss.")

    ####################################################
    # PREDICTION SANITY CHECK (VERY IMPORTANT)
    ####################################################

    preds = model.predict(X_test[:500])

    if np.std(preds) < 1e-5:
        raise RuntimeError("LSTM collapsed — constant predictions.")

    return model, scalers, val_loss


############################################################
# MAIN — GOVERNED CLOCK
############################################################

def main(start_date=None, end_date=None):

    configure_runtime()

    if start_date is None or end_date is None:
        start_date, end_date = MarketTime.window_for("lstm")

    print(f"Institutional LSTM Training | {start_date} -> {end_date}")

    df, surviving = load_data(start_date, end_date)

    df = compute_returns(df)

    dataset_hash = MetadataManager.fingerprint_dataset(
        df[["ticker", "date", "return"]]
    )

    model, scalers, val_loss = time_series_validation(df)

    atomic_save_model(model, MODEL_PATH)
    atomic_joblib_dump(scalers, SCALER_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="lstm_return_forecast",
        metrics={"val_loss": val_loss},
        features=["close_sequence"],
        training_start=start_date,
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="sequence_manifest_v1",
        extra_fields={
            "training_universe": surviving,
            "universe_hash": MetadataManager.hash_list(surviving),
            "lookback_window": LOOKBACK_WINDOW,
            "target": "log_returns"
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
