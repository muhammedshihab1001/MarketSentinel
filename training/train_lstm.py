import os
import datetime
import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from core.data.data_fetcher import StockPriceFetcher
from core.artifacts.metadata_manager import MetadataManager
from models.lstm_model import build_lstm_model


MODEL_DIR = "artifacts/lstm"
MODEL_PATH = f"{MODEL_DIR}/lstm_price_forecast.h5"
SCALER_PATH = f"{MODEL_DIR}/lstm_scaler.pkl"
METADATA_PATH = f"{MODEL_DIR}/metadata.json"

LOOKBACK_WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32


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

    prices = df[["close"]].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    return scaled, scaler, end_date


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
# TRAIN
# ---------------------------------------------------

def train():

    scaled, scaler, end_date = load_data()

    X, y = create_sequences(scaled, LOOKBACK_WINDOW)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((LOOKBACK_WINDOW, 1))

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test)
    )

    val_loss = float(history.history["val_loss"][-1])

    return model, scaler, val_loss, end_date


# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":

    model, scaler, val_loss, end_date = train()

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = MetadataManager.create_metadata(
        model_name="lstm_price_forecast",
        metrics={"val_loss": val_loss},
        features=["close_sequence"],
        training_start="2018-01-01",
        training_end=end_date
    )

    MetadataManager.save_metadata(metadata, METADATA_PATH)

    print("LSTM model saved")
    print("Metadata saved")
