"""Architecture tests.

Deliberately tiny: these check that the models are built correctly and, above
all, that they are *causal*. Whether a network forecasts well is a question for
an experiment, not a unit test.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqf.errors import ModelError
from aqf.models.deep import ARCHITECTURES, build_model, train_model

pytest.importorskip("tensorflow", reason="TensorFlow is needed for the deep models")


@pytest.fixture
def tiny_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 12, 3)).astype(np.float32)
    y = (X[:, -1, 0] * 0.8 + rng.normal(0, 0.1, 120)).astype(np.float32)
    return X[:90], y[:90], X[90:], y[90:]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_builds_and_outputs_one_value(architecture: str) -> None:
    model = build_model(architecture, (12, 3), units=8, filters=8)
    assert model.output_shape == (None, 1)
    assert model.count_params() > 0


def test_capacity_is_comparable_across_architectures() -> None:
    """A comparison where one model is ten times larger measures size, not design."""
    counts = {
        a: build_model(a, (24, 12), units=64, filters=64).count_params() for a in ARCHITECTURES
    }
    assert max(counts.values()) / min(counts.values()) < 3.0


@pytest.mark.parametrize("architecture", ["cnn", "cnn_lstm"])
def test_convolutions_are_causal(architecture: str) -> None:
    """Changing the last timestep must not change an earlier one's activation.

    'same' padding lets a convolution read the step after the one it produces.
    On a forecasting problem that is leakage, and it looks like a good result.
    """
    import tensorflow as tf

    model = build_model(architecture, (16, 2), units=8, filters=8, kernel_size=3, dropout=0.0)
    conv = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D))
    probe = tf.keras.Model(inputs=model.inputs, outputs=conv.output)

    rng = np.random.default_rng(5)
    base = rng.normal(size=(1, 16, 2)).astype(np.float32)
    altered = base.copy()
    altered[0, -1, :] += 10.0

    before = probe.predict(base, verbose=0)
    after = probe.predict(altered, verbose=0)

    # Every step but the last is unaffected by a change to the final input.
    np.testing.assert_allclose(before[0, :-1], after[0, :-1], atol=1e-5)
    assert not np.allclose(before[0, -1], after[0, -1], atol=1e-5)


def test_an_unknown_architecture_is_refused() -> None:
    with pytest.raises(ModelError, match="unknown architecture"):
        build_model("transformer", (12, 3))  # type: ignore[arg-type]


def test_a_kernel_wider_than_the_window_is_refused() -> None:
    with pytest.raises(ModelError, match="exceeds the lookback"):
        build_model("cnn", (2, 3), kernel_size=5)


def test_the_same_seed_builds_the_same_weights() -> None:
    first = build_model("gru", (12, 3), units=8, seed=99)
    second = build_model("gru", (12, 3), units=8, seed=99)
    for a, b in zip(first.get_weights(), second.get_weights(), strict=True):
        np.testing.assert_allclose(a, b)


@pytest.mark.slow
def test_training_reduces_the_loss_and_records_what_happened(
    tiny_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, y_train, X_validation, y_validation = tiny_data
    trained = train_model(
        "gru",
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=6,
        units=8,
        patience=6,
        batch_size=32,
    )
    assert trained.epochs_run > 0
    assert trained.train_loss[-1] < trained.train_loss[0]
    assert trained.parameters > 0
    assert trained.fit_seconds > 0.0
    assert trained.predict(X_validation).shape == (len(X_validation),)


@pytest.mark.slow
def test_early_stopping_restores_the_best_weights(
    tiny_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Without restoration the reported model is whatever the last epoch left."""
    X_train, y_train, X_validation, y_validation = tiny_data
    trained = train_model(
        "gru",
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=40,
        units=8,
        patience=3,
        batch_size=32,
    )
    if trained.stopped_early:
        best = min(trained.validation_loss)
        restored = float(trained.model.evaluate(X_validation, y_validation, verbose=0))
        assert restored == pytest.approx(best, rel=0.05)


def test_training_without_windows_is_refused() -> None:
    empty = np.zeros((0, 12, 3), dtype=np.float32)
    with pytest.raises(ModelError, match="must both contain windows"):
        train_model("gru", empty, np.zeros(0), empty, np.zeros(0))
