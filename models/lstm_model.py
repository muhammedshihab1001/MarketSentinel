import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np


def build_lstm_model(input_shape):
    """
    Institutional-grade LSTM architecture.

    Design goals:
    - stable gradients
    - low overfitting
    - CPU-safe inference
    - reproducible training
    """

    model = Sequential([
        LSTM(
            128,
            return_sequences=True,
            input_shape=input_shape,
            dropout=0.2,
            recurrent_dropout=0.1
        ),
        LSTM(
            64,
            dropout=0.2,
            recurrent_dropout=0.1
        ),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    optimizer = Adam(
        learning_rate=0.0005,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber()
    )

    return model


def forecast_lstm(model, scaler, recent_prices, horizon=7):
    """
    Rolling autoregressive forecast.

    Safety features:
    - dtype enforcement
    - shape protection
    - scaler consistency
    """

    if len(recent_prices) < 60:
        raise ValueError("At least 60 prices required for LSTM forecast")

    scaled_seq = scaler.transform(
        recent_prices.astype("float32")
    )[-60:]

    seq = scaled_seq.reshape(1, 60, 1).astype("float32")

    preds = []

    for _ in range(horizon):

        pred_scaled = model.predict(seq, verbose=0)[0][0]

        preds.append(pred_scaled)

        next_step = np.array(pred_scaled, dtype="float32").reshape(1, 1, 1)

        seq = np.concatenate(
            [seq[:, 1:, :], next_step],
            axis=1
        )

    preds = scaler.inverse_transform(
        np.array(preds, dtype="float32").reshape(-1, 1)
    ).flatten()

    return preds.tolist()
