import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    LayerNormalization
)
from tensorflow.keras.optimizers import Adam
import numpy as np


# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

def build_lstm_model(input_shape):
    """
    Institutional-grade LSTM.

    Guarantees:
    - CuDNN-compatible kernels
    - stabilized gradients
    - deterministic-friendly
    - low inference latency
    """

    model = Sequential([

        LSTM(
            128,
            return_sequences=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal"
        ),

        LayerNormalization(),
        Dropout(0.2),

        LSTM(
            64,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal"
        ),

        LayerNormalization(),
        Dropout(0.2),

        Dense(
            32,
            activation="gelu",
            kernel_initializer="glorot_uniform"
        ),

        Dense(1)
    ])

    optimizer = Adam(
        learning_rate=3e-4,     # safer than 1e-3
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=0.5)
    )

    return model


# ---------------------------------------------------
# FORECAST
# ---------------------------------------------------

def forecast_lstm(
    model,
    scaler,
    recent_prices,
    lookback=60,
    horizon=7
):
    """
    Rolling autoregressive forecast.

    Guarantees:
    - dtype enforcement
    - finite predictions
    - scaler consistency
    """

    if len(recent_prices) < lookback:
        raise ValueError(
            f"At least {lookback} prices required for LSTM forecast"
        )

    prices = recent_prices.astype("float32").reshape(-1, 1)

    scaled_seq = scaler.transform(prices)[-lookback:]

    seq = scaled_seq.reshape(1, lookback, 1).astype("float32")

    preds = []

    for _ in range(horizon):

        pred_scaled = model.predict(seq, verbose=0)[0][0]

        if not np.isfinite(pred_scaled):
            raise RuntimeError(
                "LSTM produced non-finite prediction."
            )

        preds.append(pred_scaled)

        next_step = np.array(
            pred_scaled,
            dtype="float32"
        ).reshape(1, 1, 1)

        seq = np.concatenate(
            [seq[:, 1:, :], next_step],
            axis=1
        )

    preds = scaler.inverse_transform(
        np.array(preds, dtype="float32").reshape(-1, 1)
    ).flatten()

    if not np.isfinite(preds).all():
        raise RuntimeError(
            "Inverse-scaled predictions contain invalid values."
        )

    return preds.tolist()
