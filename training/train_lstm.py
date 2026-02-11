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
from training.backtesting.walk_forward import WalkForwardValidator
from models.lstm_model import build_lstm_model


MODEL_DIR = "artifacts/lstm"

TEMP_MODEL_PATH = f"{MODEL_DIR}/model.keras"
TEMP_SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
TEMP_METADATA_PATH = f"{MODEL_DIR}/metadata.json"

LOOKBACK_WINDOW = 60
EPOCHS = 30
BATCH_SIZE = 32

MIN_ROWS = 1500
MIN_SHARPE = 0.25
MAX_DRAWDOWN = -0.40

SEED = 42


TRAINING_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"
]


# ---------------------------------------------------
# DETERMINISM
# ---------------------------------------------------

def set_seeds():

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    tf.config.set_visible_devices([], "GPU")

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


# ---------------------------------------------------
# CRASH-SAFE MODEL SAVE
# ---------------------------------------------------

def atomic_save_model(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_dir = path + "_tmp"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    model.save(tmp_dir)

    if os.path.exists(path):
        shutil.rmtree(path)

    os.replace(tmp_dir, path)


# ---------------------------------------------------
# DATA
# ---------------------------------------------------

def load_data():

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
        raise RuntimeError("No datasets fetched for LSTM.")

    df = pd.concat(datasets, ignore_index=True)
    df = df.sort_values(["ticker", "date"])

    if len(df) < MIN_ROWS:
        raise RuntimeError(
            f"LSTM training aborted — insufficient rows ({len(df)})"
        )

    return df.reset_index(drop=True), end_date


# ---------------------------------------------------

def create_sequences(data, lookback):

    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])

    return np.array(X), np.array(y)


# ---------------------------------------------------
# WINDOW TRAINER (Walk Forward)
# ---------------------------------------------------

def train_window_model(train_df):

    prices = train_df["close"].astype("float32").values.reshape(-1, 1)

    if not np.isfinite(prices).all():
        raise RuntimeError("Non-finite values detected in price series.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = create_sequences(scaled, LOOKBACK_WINDOW)

    model = build_lstm_model((LOOKBACK_WINDOW, 1))

    model.fit(
        X,
        y,
        epochs=6,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    return model, scaler


# ---------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------

def generate_signals(model_scaler, test_df):

    model, scaler = model_scaler

    prices = test_df["close"].astype("float32").values.reshape(-1, 1)

    if not np.isfinite(prices).all():
        raise RuntimeError("Non-finite values detected in price series.")

    scaled = scaler.transform(prices)

    signals = []

    for i in range(LOOKBACK_WINDOW, len(scaled)):

        seq = scaled[i-LOOKBACK_WINDOW:i].reshape(1, LOOKBACK_WINDOW, 1)

        pred_scaled = model.predict(seq, verbose=0)[0][0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]

        current_price = prices[i-1][0]

        signals.append(
            "BUY" if pred_price > current_price else "SELL"
        )

    return ["HOLD"] * LOOKBACK_WINDOW + signals


# ---------------------------------------------------
# FULL TRAIN
# ---------------------------------------------------

def train_full_model(df):

    prices = df["close"].astype("float32").values.reshape(-1, 1)

    if not np.isfinite(prices).all():
        raise RuntimeError("Non-finite values detected in price series.")

    split = int(len(prices) * 0.8)

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(prices[:split])
    test_scaled = scaler.transform(prices[split:])

    X_train, y_train = create_sequences(train_scaled, LOOKBACK_WINDOW)
    X_test, y_test = create_sequences(test_scaled, LOOKBACK_WINDOW)

    model = build_lstm_model((LOOKBACK_WINDOW, 1))

    early = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early],
        verbose=1
    )

    val_loss = float(min(history.history["val_loss"]))

    return model, scaler, val_loss


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":

    print("Institutional LSTM Training")

    set_seeds()

    df, end_date = load_data()

    # HASH ONLY WHAT MODEL LEARNS FROM
    dataset_hash = MetadataManager.fingerprint_dataset(
        df[["close"]]
    )

    wf = WalkForwardValidator(
        model_trainer=train_window_model,
        signal_generator=generate_signals
    )

    strategy_metrics = wf.run(df)

    if strategy_metrics["avg_sharpe"] < MIN_SHARPE:
        raise RuntimeError("LSTM rejected — Sharpe too low")

    if strategy_metrics["max_drawdown"] < MAX_DRAWDOWN:
        raise RuntimeError("LSTM rejected — drawdown too high")

    model, scaler, val_loss = train_full_model(df)

    atomic_save_model(model, TEMP_MODEL_PATH)
    joblib.dump(scaler, TEMP_SCALER_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="lstm_price_forecast",
        metrics={"val_loss": val_loss, **strategy_metrics},
        features=["close_sequence"],
        training_start="2018-01-01",
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
        os.path.join(MODEL_DIR, version, "scaler.pkl")
    )

    print(f"LSTM registered → {version}")
