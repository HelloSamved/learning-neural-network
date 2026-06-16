"""A small MNIST neural network implemented only with NumPy."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


LAYER_SIZES = (784, 28, 28, 10)


def initialize_parameters(seed=42):
    """Initialize a 784-28-28-10 network using He initialization."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((28, 784)) * np.sqrt(2 / 784)
    b1 = np.zeros((28, 1))
    w2 = rng.standard_normal((28, 28)) * np.sqrt(2 / 28)
    b2 = np.zeros((28, 1))
    w3 = rng.standard_normal((10, 28)) * np.sqrt(2 / 28)
    b3 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3
2

def relu(values):
    return np.maximum(values, 0)


def softmax(values):
    shifted = values - np.max(values, axis=0, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=0, keepdims=True)


def forward_propagation(parameters, inputs):
    w1, b1, w2, b2, w3, b3 = parameters
    z1 = w1 @ inputs + b1
    a1 = relu(z1)
    z2 = w2 @ a1 + b2
    a2 = relu(z2)
    z3 = w3 @ a2 + b3
    probabilities = softmax(z3)
    return z1, a1, z2, a2, z3, probabilities


def _backward_propagation(parameters, inputs, labels, cache):
    w1, _, w2, _, w3, _ = parameters
    z1, a1, z2, a2, _, probabilities = cache
    batch_size = inputs.shape[1]

    expected = np.zeros_like(probabilities)
    expected[labels, np.arange(batch_size)] = 1

    dz3 = probabilities - expected
    dw3 = dz3 @ a2.T / batch_size
    db3 = np.sum(dz3, axis=1, keepdims=True) / batch_size
    dz2 = (w3.T @ dz3) * (z2 > 0)
    dw2 = dz2 @ a1.T / batch_size
    db2 = np.sum(dz2, axis=1, keepdims=True) / batch_size
    dz1 = (w2.T @ dz2) * (z1 > 0)
    dw1 = dz1 @ inputs.T / batch_size
    db1 = np.sum(dz1, axis=1, keepdims=True) / batch_size
    return dw1, db1, dw2, db2, dw3, db3


def predict_proba(inputs, parameters):
    """Return digit probabilities for normalized, flattened image columns."""
    return forward_propagation(parameters, inputs)[-1]


def predict(inputs, parameters):
    return np.argmax(predict_proba(inputs, parameters), axis=0)


def accuracy(inputs, labels, parameters):
    return float(np.mean(predict(inputs, parameters) == labels))


def train_model(
    inputs,
    labels,
    epochs=24,
    learning_rate=0.08,
    batch_size=256,
    seed=42,
):
    """Train with mini-batch gradient descent and return parameters and history."""
    parameters = initialize_parameters(seed)
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(epochs):
        order = rng.permutation(inputs.shape[1])
        for start in range(0, inputs.shape[1], batch_size):
            batch = order[start : start + batch_size]
            batch_inputs = inputs[:, batch]
            batch_labels = labels[batch]
            cache = forward_propagation(parameters, batch_inputs)
            gradients = _backward_propagation(
                parameters, batch_inputs, batch_labels, cache
            )
            parameters = tuple(
                parameter - learning_rate * gradient
                for parameter, gradient in zip(parameters, gradients)
            )
        history.append(accuracy(inputs, labels, parameters))

    return parameters, history


def load_dataset(csv_path, limit=None):
    frame = pd.read_csv(csv_path, nrows=limit)
    labels = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    inputs = frame.iloc[:, 1:].to_numpy(dtype=np.float64).T / 255.0
    return inputs, labels


def save_model(path, parameters, **metadata):
    values = {f"parameter_{index}": value for index, value in enumerate(parameters)}
    values.update(metadata)
    np.savez_compressed(path, **values)


def load_model(path):
    model = np.load(path)
    return tuple(model[f"parameter_{index}"] for index in range(6))


def preprocess_image(image):
    """Convert an uploaded image into an MNIST-like 28x28 normalized image."""
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.mean() > 127:
        image = ImageOps.invert(image)

    bounding_box = image.getbbox()
    if bounding_box:
        image = image.crop(bounding_box)

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - image.width) // 2, (28 - image.height) // 2)
    canvas.paste(image, offset)
    normalized = np.asarray(canvas, dtype=np.float64) / 255.0
    return canvas, normalized.reshape(784, 1)


def train_and_save(csv_path, output_path, epochs=24):
    """Train a deployable model from train.csv and save its weights."""
    inputs, labels = load_dataset(csv_path)
    rng = np.random.default_rng(42)
    order = rng.permutation(inputs.shape[1])
    validation = order[:2000]
    training = order[2000:]

    parameters, history = train_model(
        inputs[:, training], labels[training], epochs=epochs
    )
    validation_accuracy = accuracy(
        inputs[:, validation], labels[validation], parameters
    )
    save_model(
        output_path,
        parameters,
        validation_accuracy=np.array(validation_accuracy),
        training_accuracy=np.array(history[-1]),
        epochs=np.array(epochs),
    )
    return history[-1], validation_accuracy


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    train_accuracy, validation_accuracy = train_and_save(
        project_dir / "train.csv", project_dir / "mnist_weights.npz"
    )
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation accuracy: {validation_accuracy:.2%}")
