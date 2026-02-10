import os
import datetime
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import shutil
import tempfile

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from core.artifacts.model_registry import ModelRegistry
from training.backtesting.walk_forward import WalkForwardValidator
from models.lstm_model import build_lstm_model


MODEL_DIR = "artifacts/lstm"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.tmp.keras"
FINAL_MODEL_NAME = "model.keras"

TEMP_SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

LOOKBACK_WINDOW = 60
EPOCHS = 40
BATCH_SIZE = 32

MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40

SEED = 42


# ---------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------

def set_seeds():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    tf.config.set_visible_devices([], "GPU")

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


# ---------------------------------------------------
# DATA
# ---------------------------------------------------

def load_data():

    end_date = datetime.date.today().isoformat()

    fetcher = StockPriceFetcher()

    df = fetcher.fetch(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date=end_date
    )

    if df.empty:
        raise ValueError("No price data fetched for LSTM training")

    return df, end_date


# ---------------------------------------------------
# SEQUENCE BUILDER
# ---------------------------------------------------

def create_sequences(data, lookback):

    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])

    return np.array(X), np.array(y)


# ---------------------------------------------------
# TRAIN SINGLE WINDOW
# ---------------------------------------------------

def train_model(train_prices):

    scaler = MinMaxScaler()

    scaled_train = scaler.fit_transform(
        train_prices.reshape(-1, 1)
    )

    X, y = create_sequences(scaled_train, LOOKBACK_WINDOW)

    model = build_lstm_model((LOOKBACK_WINDOW, 1))

    early_stop = EarlyStopping(
        monitor="loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=10,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop]
    )

    return model, scaler


# ---------------------------------------------------
# SIGNAL GENERATOR
# ---------------------------------------------------

def generate_signals(model_scaler_tuple, test_df):

    model, scaler = model_scaler_tuple

    prices = test_df["close"].values.reshape(-1, 1)
    scaled = scaler.transform(prices)

    signals = []

    for i in range(LOOKBACK_WINDOW, len(scaled)):

        seq = scaled[i-LOOKBACK_WINDOW:i].reshape(1, LOOKBACK_WINDOW, 1)

        pred = model.predict(seq, verbose=0)[0][0]

        if pred > scaled[i-1]:
            signals.append("BUY")
        else:
            signals.append("SELL")

    signals = ["HOLD"] * LOOKBACK_WINDOW + signals

    return signals


# ---------------------------------------------------
# FINAL TRAIN
# ---------------------------------------------------

def train_full_model(df):

    prices = df["close"].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1))

    X, y = create_sequences(scaled, LOOKBACK_WINDOW)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((LOOKBACK_WINDOW, 1))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    val_loss = float(min(model.history.history["val_loss"]))

    return model, scaler, val_loss


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":

    print("\nInstitutional LSTM Training\n")

    set_seeds()

    df, end_date = load_data()

    dataset_hash = MetadataManager.fingerprint_dataset(df)

    # ---------------------------------------------------
    # WALK FORWARD
    # ---------------------------------------------------

    wf = WalkForwardValidator(
        model_trainer=lambda d: train_model(d["close"].values),
        signal_generator=generate_signals
    )

    strategy_metrics = wf.run(df)

    if strategy_metrics["avg_sharpe"] < MIN_SHARPE:
        raise RuntimeError("LSTM rejected — Sharpe too low")

    if strategy_metrics["max_drawdown"] < MAX_DRAWDOWN:
        raise RuntimeError("LSTM rejected — drawdown too high")

    print("Strategy passed governance.\n")

    # ---------------------------------------------------
    # FINAL TRAIN
    # ---------------------------------------------------

    model, scaler, val_loss = train_full_model(df)

    # SAFE SAVE
    model.save(TEMP_MODEL_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="lstm_price_forecast",
        metrics={
            "val_loss": val_loss,
            **strategy_metrics
        },
        features=["close_sequence"],
        training_start="2018-01-01",
        training_end=end_date,
        dataset_hash=dataset_hash
    )

    joblib.dump(scaler, TEMP_SCALER_PATH)
    MetadataManager.save_metadata(metadata, TEMP_METADATA_PATH)

    version_dir = ModelRegistry.register_model(
        MODEL_DIR,
        TEMP_MODEL_PATH,
        TEMP_METADATA_PATH
    )

    shutil.move(
        TEMP_SCALER_PATH,
        os.path.join(version_dir, "scaler.pkl")
    )

    print("LSTM registered.")
    print(f"Version: {version_dir}")
