import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

def forecast_lstm(model, scaler, recent_prices, horizon=7):
    """
    Predict future prices using rolling LSTM forecast
    """
    seq = scaler.transform(recent_prices)[-60:].reshape(1, 60, 1)
    preds = []

    for _ in range(horizon):
        pred = model.predict(seq, verbose=0)[0][0]
        preds.append(pred)
        seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)

    preds = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

    return preds.tolist()