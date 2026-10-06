from pathlib import Path
import time

from data.digit_dataset_loader import load_dataset

import numpy as np

from models.activation import Tanh
from models.neural_network import MultiLayerPerceptron
from models.optimization import Optimization

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ((64,), (128, 64))
LEARNING_RATES = (0.1, 0.05, 0.01, 0.001, 0.0001)
OPTIMIZERS = ("None", "Momentum", "RMSProp", "Adam")

def classification_report(predicted, labels):
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    per_digit, f1_values = {}, []
    for digit in range(10):
        tp = confusion[digit, digit]
        actual = confusion[digit].sum()
        positive = confusion[:, digit].sum()
        f1 = float(2 * tp / (actual + positive)) if actual + positive else None
        if f1 is not None:
            f1_values.append(f1)
        per_digit[str(digit)] = {
            "support": int(actual),
            "precision": float(tp / positive) if positive else 0.0,
            "recall": float(tp / actual) if actual else None,
            "f1": f1,
        }
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "macro_f1": float(np.mean(f1_values)),
        "per_digit": per_digit,
        "confusion_matrix": confusion.tolist(),
    }

def targets(labels):
    encoded = np.full((len(labels), 10), -1.0)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded

def read_digits(path: Path):
    frame = load_dataset(str(path))
    images = np.stack(frame["image"].to_numpy()).astype(np.float32) # type: ignore
    labels = frame["label"].to_numpy(dtype=np.int64)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"{path}: each image must contain 784 pixels")
    if not np.isfinite(images).all() or np.any((labels < 0) | (labels > 9)):
        raise ValueError(f"{path}: invalid pixels or labels")
    return images, labels

def stratified_split(labels, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for digit in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == digit))
        count = max(1, round(len(indices) * validation_fraction))
        if count == len(indices):
            raise ValueError(f"digit {digit} has too few samples to split")
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return np.array(train), np.array(validation)

def evaluate(model, images, labels, tracked_digits=(), chunk_size=2048):
    correct, squared_error, output_count = 0, 0.0, 0
    digit_hits = {digit: 0 for digit in tracked_digits}
    digit_counts = {
        digit: int(np.count_nonzero(labels == digit)) for digit in tracked_digits
    }
    for start in range(0, len(labels), chunk_size):
        end = start + chunk_size
        outputs = model.classify(images[start:end])
        predictions = outputs.argmax(axis=1)
        batch_labels = labels[start:end]
        correct += int(np.count_nonzero(predictions == batch_labels))
        squared_error += float(np.sum((outputs - targets(batch_labels)) ** 2))
        output_count += outputs.size
        for digit in tracked_digits:
            digit_hits[digit] += int(
                np.count_nonzero((predictions == digit) & (batch_labels == digit))
            )
    metrics = {
        "accuracy": correct / len(labels),
        "mse": squared_error / output_count,
    }
    for digit in tracked_digits:
        metrics[f"recall_{digit}"] = (
            digit_hits[digit] / digit_counts[digit] if digit_counts[digit] else None
        )
    return metrics


def make_model(topology, learning_rate, optimizer_name, seed, output_dir, label):
    np.random.seed(seed)
    optimizer = Optimization.from_string(optimizer_name)
    return MultiLayerPerceptron(
        list(topology), Tanh(), learning_rate, optimization=optimizer, save=output_dir, model_name=label
    )


def fit_model(
    training_data,
    topology,
    learning_rate,
    optimizer_name,
    batch_size,
    epochs,
    seed,
    output_dir,
    label
):
    model = make_model(topology, learning_rate, optimizer_name, seed, output_dir, label)
    started = time.perf_counter()
    model.train(
        training_data, epochs=epochs, epsilon=-1, batch_size=batch_size, seed=seed
    )
    return model, time.perf_counter() - started

