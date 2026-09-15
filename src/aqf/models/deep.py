"""Deep sequence models: LSTM, GRU, CNN and a CNN-LSTM hybrid.

The four architectures the dissertation compares, rebuilt so the comparison is
fair: identical windows, identical scaling, identical training budget, identical
early stopping, and one seed controlling all of it. Differences that survive that
are differences between architectures rather than between runs.

What each is supposed to contribute:

*LSTM* and *GRU* carry state across the window, so they can in principle use
something that happened twenty hours ago. GRU has fewer parameters for the same
job, which matters on a series this short.

*CNN* has no memory at all. It convolves over the window looking for local
shapes - a sharp rise, a plateau - and is fast. On a strongly autocorrelated
series it often does as well as the recurrent models, which is informative
rather than disappointing.

*CNN-LSTM* puts the convolution first as a feature extractor and the recurrence
second. It is the architecture most likely to win on paper and the one most
likely to be over-parameterised for seven thousand training windows.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..errors import ModelError
from ..logging_utils import get_logger

logger = get_logger(__name__)

Architecture = Literal["lstm", "gru", "cnn", "cnn_lstm"]
ARCHITECTURES: tuple[Architecture, ...] = ("lstm", "gru", "cnn", "cnn_lstm")


@dataclass
class TrainedModel:
    """A fitted network and what it cost to fit."""

    name: str
    model: Any
    epochs_run: int
    best_epoch: int
    fit_seconds: float
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    parameters: int = 0

    @property
    def stopped_early(self) -> bool:
        return self.best_epoch < self.epochs_run - 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X, verbose=0)).reshape(-1)

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "name": self.name,
            "parameters": float(self.parameters),
            "epochs_run": float(self.epochs_run),
            "best_epoch": float(self.best_epoch),
            "fit_seconds": self.fit_seconds,
            "stopped_early": self.stopped_early,
            "final_train_loss": self.train_loss[-1] if self.train_loss else float("nan"),
            "best_validation_loss": (
                min(self.validation_loss) if self.validation_loss else float("nan")
            ),
        }


def configure_threads(n_threads: int | None = None) -> None:
    """Bound TensorFlow's thread use.

    Left alone it takes every core, which makes a timing measurement on a shared
    machine meaningless and lets two concurrent runs contend into nonsense.
    """
    threads = n_threads or int(os.environ.get("AQF_NUM_THREADS", "0") or 0)
    if threads <= 0:
        return
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)


def build_model(
    architecture: Architecture,
    input_shape: tuple[int, int],
    *,
    units: int = 64,
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    seed: int = 20260101,
) -> Any:
    """Construct one architecture.

    Capacity is deliberately modest and identical across the four. A sweep that
    gives one architecture more units than another measures the sweep, not the
    architecture.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(seed)
    lookback, n_features = input_shape
    if lookback < 1 or n_features < 1:
        raise ModelError(f"invalid input shape {input_shape}")

    inputs = layers.Input(shape=input_shape)

    if architecture == "lstm":
        x = layers.LSTM(units, return_sequences=False)(inputs)
    elif architecture == "gru":
        x = layers.GRU(units, return_sequences=False)(inputs)
    elif architecture == "cnn":
        if kernel_size > lookback:
            raise ModelError(f"kernel_size {kernel_size} exceeds the lookback of {lookback}")
        x = layers.Conv1D(filters, kernel_size, activation="relu", padding="causal")(inputs)
        x = layers.Conv1D(filters, kernel_size, activation="relu", padding="causal")(x)
        x = layers.GlobalAveragePooling1D()(x)
    elif architecture == "cnn_lstm":
        if kernel_size > lookback:
            raise ModelError(f"kernel_size {kernel_size} exceeds the lookback of {lookback}")
        # Causal padding, so a convolution never sees a step after the one it
        # is producing. 'same' padding leaks the future into every window.
        x = layers.Conv1D(filters, kernel_size, activation="relu", padding="causal")(inputs)
        x = layers.LSTM(units, return_sequences=False)(x)
    else:
        raise ModelError(f"unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")

    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_model(
    architecture: Architecture,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 8,
    units: int = 64,
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    seed: int = 20260101,
    n_threads: int | None = None,
) -> TrainedModel:
    """Train one architecture with early stopping on validation loss.

    Early stopping restores the best weights rather than keeping the last ones.
    Without that, the reported model is whatever the final epoch produced, which
    on a small series is frequently worse than something six epochs earlier.
    """
    import tensorflow as tf

    configure_threads(n_threads)
    if len(X_train) == 0 or len(X_validation) == 0:
        raise ModelError("training and validation sets must both contain windows")

    model = build_model(
        architecture,
        (X_train.shape[1], X_train.shape[2]),
        units=units,
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        learning_rate=learning_rate,
        seed=seed,
    )

    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, mode="min"
    )
    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[stopper],
        shuffle=False,  # windows are ordered in time; shuffling batches is fine
        verbose=0,
    )
    fit_seconds = time.perf_counter() - started

    validation_loss = [float(v) for v in history.history.get("val_loss", [])]
    train_loss = [float(v) for v in history.history.get("loss", [])]
    best_epoch = int(np.argmin(validation_loss)) if validation_loss else 0

    trained = TrainedModel(
        name=architecture,
        model=model,
        epochs_run=len(train_loss),
        best_epoch=best_epoch,
        fit_seconds=fit_seconds,
        train_loss=train_loss,
        validation_loss=validation_loss,
        parameters=int(model.count_params()),
    )
    logger.info(
        "model_trained",
        architecture=architecture,
        parameters=trained.parameters,
        epochs_run=trained.epochs_run,
        best_epoch=best_epoch,
        stopped_early=trained.stopped_early,
        fit_seconds=round(fit_seconds, 2),
        best_val_loss=round(min(validation_loss), 6) if validation_loss else None,
    )
    return trained
