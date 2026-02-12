import os
import datetime
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from models.lstm_model import build_lstm_model


MODEL_DIR = "artifacts/lstm"

MODEL_PATH = f"{MODEL_DIR}/model.keras"
SCALER_PATH = f"{MODEL_DIR}/scalers.pkl"
METADATA_PATH = f"{MODEL_DIR}/metadata.json"

LOOKBACK_WINDOW = 60
EPOCHS = 50
MIN_ROWS_PER_TICKER = 800
SEED = 42


TRAINING_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
    "JPM","GS","BAC","LLY","XOM","AVGO","AMD",
    "SPY","QQQ","IWM"
]


############################################
# RUNTIME CONFIG
############################################

GPU_AVAILABLE = False


def configure_runtime():

    global GPU_AVAILABLE

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")

    GPU_AVAILABLE = len(gpus) > 0

    if GPU_AVAILABLE:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print(f"GPU detected: {gpus[0].name}")

        except RuntimeError as e:
            print(f"GPU configuration failed. Falling back to CPU: {e}")
            GPU_AVAILABLE = False
    else:
        print("Running on CPU.")


############################################

def get_batch_size():
    return 128 if GPU_AVAILABLE else 32


############################################
# TRUE ATOMIC SAVE
############################################

def atomic_save_model(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = path + ".tmp.keras"

    model.save(tmp_path)

    os.replace(tmp_path, path)


############################################

def load_data():

    fetcher = StockPriceFetcher()
    end_date = datetime.date.today().isoformat()

    datasets = []

    for ticker in TRAINING_TICKERS:

        df = fetcher.fetch(
            ticker=ticker,
            start_date="2012-01-01",
            end_date=end_date
        )

        if df.empty or len(df) < MIN_ROWS_PER_TICKER:
            continue

        df["ticker"] = ticker
        datasets.append(df)

    if not datasets:
        raise RuntimeError("No datasets fetched for LSTM.")

    df = pd.concat(datasets, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["ticker", "date"], inplace=True)

    return df.reset_index(drop=True), end_date


############################################

def build_sequences(df):

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

    if not X_all:
        raise RuntimeError("No sequences generated.")

    return (
        np.asarray(X_all, dtype=np.float32),
        np.asarray(y_all, dtype=np.float32),
        scalers
    )


############################################

def time_series_validation(df):

    split_date = df["date"].quantile(0.8)

    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    X_train, y_train, scalers = build_sequences(train_df)
    X_test, y_test, _ = build_sequences(test_df)

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
        batch_size=get_batch_size(),
        callbacks=[early],
        verbose=1
    )

    val_loss = float(min(history.history["val_loss"]))

    return model, scalers, val_loss


############################################

if __name__ == "__main__":

    print("Institutional LSTM Training")

    configure_runtime()

    df, end_date = load_data()

    dataset_hash = MetadataManager.fingerprint_dataset(
        df[["ticker", "date", "close"]]
    )

    model, scalers, val_loss = time_series_validation(df)

    atomic_save_model(model, MODEL_PATH)
    joblib.dump(scalers, SCALER_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="lstm_price_forecast",
        metrics={"val_loss": val_loss},
        features=["close_sequence"],
        training_start="2012-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="sequence"
    )

    MetadataManager.save_metadata(metadata, METADATA_PATH)

    version = ModelRegistry.register_model(
        MODEL_DIR,
        MODEL_PATH,
        METADATA_PATH
    )

    os.replace(
        SCALER_PATH,
        os.path.join(MODEL_DIR, version, "scalers.pkl")
    )

    print(f"LSTM registered → {version}")
