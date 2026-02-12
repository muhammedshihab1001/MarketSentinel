import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np


def _configure_runtime():

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        except Exception:
            pass


_configure_runtime()


def build_lstm_model(input_shape):

    model = Sequential([

        LSTM(
            64,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal"
        ),

        LayerNormalization(),
        Dropout(0.15),

        LSTM(
            32,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal"
        ),

        LayerNormalization(),
        Dropout(0.15),

        Dense(
            16,
            activation="gelu",
            kernel_initializer="glorot_uniform"
        ),

        Dense(1, dtype="float32")
    ])

    optimizer = Adam(
        learning_rate=3e-4,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=0.5)
    )

    return model


def forecast_lstm(
    model,
    scaler,
    recent_prices,
    lookback=60,
    horizon=7
):

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
