import os
import datetime
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import shutil

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from models.lstm_model import build_lstm_model


MODEL_DIR = "artifacts/lstm"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.keras"
TEMP_SCALER_PATH = f"{MODEL_DIR}/scalers.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

LOOKBACK_WINDOW = 60
EPOCHS = 30
BATCH_SIZE = 32

MIN_ROWS_PER_TICKER = 800

SEED = 42


TRAINING_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
    "JPM","GS","BAC","LLY","XOM","AVGO","AMD",
    "SPY","QQQ","IWM"
]


############################################

def set_seeds():

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


############################################

def atomic_save_model(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_dir = path + "_tmp"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    model.save(tmp_dir)

    if os.path.exists(path):
        shutil.rmtree(path)

    os.replace(tmp_dir, path)


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
# SAFE SEQUENCE BUILDER
############################################

def build_sequences(df):

    X_all = []
    y_all = []
    scalers = {}

    for ticker, tdf in df.groupby("ticker"):

        prices = tdf["close"].astype("float32").values.reshape(-1,1)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)

        scalers[ticker] = scaler

        for i in range(len(scaled) - LOOKBACK_WINDOW):

            X_all.append(scaled[i:i+LOOKBACK_WINDOW])
            y_all.append(scaled[i+LOOKBACK_WINDOW])

    if not X_all:
        raise RuntimeError("No sequences generated.")

    return np.array(X_all), np.array(y_all), scalers


############################################
# SIMPLE TIME SERIES VALIDATION
############################################

def time_series_validation(df):

    split_date = df["date"].quantile(0.8)

    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    X_train, y_train, scalers = build_sequences(train_df)
    X_test, y_test, _ = build_sequences(test_df)

    model = build_lstm_model((LOOKBACK_WINDOW,1))

    early = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test,y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early],
        verbose=1
    )

    val_loss = float(min(history.history["val_loss"]))

    return model, scalers, val_loss


############################################

if __name__ == "__main__":

    print("Institutional LSTM Training")

    set_seeds()

    df, end_date = load_data()

    dataset_hash = MetadataManager.fingerprint_dataset(
        df[["close"]]
    )

    model, scalers, val_loss = time_series_validation(df)

    atomic_save_model(model, TEMP_MODEL_PATH)
    joblib.dump(scalers, TEMP_SCALER_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="lstm_price_forecast",
        metrics={"val_loss": val_loss},
        features=["close_sequence"],
        training_start="2012-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash,
        metadata_type="model"
    )

    MetadataManager.save_metadata(metadata, TEMP_METADATA_PATH)

    version = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    shutil.move(
        TEMP_SCALER_PATH,
        os.path.join(MODEL_DIR, version, "scalers.pkl")
    )

    print(f"LSTM registered → {version}")
