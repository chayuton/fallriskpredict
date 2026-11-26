"""
GRU classifier.  >>> EDIT build_gru() so the architecture and optimiser match the paper exactly,
otherwise the LOSO numbers will describe a different model.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

import config as C  # noqa: E402


def build_gru(window: int, n_features: int, n_classes: int, lr: float = C.LEARNING_RATE):
    inputs = keras.Input(shape=(window, n_features))
    x = layers.GRU(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(32)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
