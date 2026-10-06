"""Study digit classification with the shared multicapa perceptron."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

# Small matrix products are usually faster with one BLAS worker per process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from data.digit_dataset_loader import load_dataset
from models.activation import Tanh
from models.neural_network import MultiLayerPerceptron
from models.optimization import Momentum, NoOptimization


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ((64,), (128, 64))
LEARNING_RATES = (0.01, 0.05)
OPTIMIZERS = ("sgd", "momentum")


def read_digits(path: Path):
    frame = load_dataset(str(path))
    images = np.stack(frame["image"].to_numpy()).astype(np.float32)
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


def targets(labels):
    encoded = np.full((len(labels), 10), -1.0)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def make_model(topology, learning_rate, optimizer_name, seed):
    np.random.seed(seed)
    optimizer = Momentum(0.9) if optimizer_name == "momentum" else NoOptimization()
    return MultiLayerPerceptron(
        list(topology), Tanh(), learning_rate, optimization=optimizer,
    )


def evaluate(model, images, labels):
    outputs = model.classify(images)
    predictions = outputs.argmax(axis=1)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "mse": float(np.mean((outputs - targets(labels)) ** 2)),
    }


def run_configuration(training_data, training_images, training_labels,
                      validation_images, validation_labels,
                      topology, learning_rate, optimizer_name, batch_size,
                      epochs, seed, mode):
    model = make_model(topology, learning_rate, optimizer_name, seed)
    started = time.perf_counter()
    model.train(training_data, epochs=epochs, epsilon=-1,
                batch_size=batch_size, seed=seed)
    seconds = time.perf_counter() - started
    train_metrics = evaluate(model, training_images, training_labels)
    metrics = evaluate(model, validation_images, validation_labels)
    result = {
        "mode": mode, "topology": list(topology),
        "learning_rate": learning_rate, "optimizer": optimizer_name,
        "batch_size": batch_size, "epochs": epochs,
        "training": train_metrics, "validation": metrics,
        "train_seconds": seconds,
    }
    print(f"{mode:6} {topology} {optimizer_name:8} lr={learning_rate}: "
          f"accuracy={metrics['accuracy']:.4f} ({seconds:.1f}s)", flush=True)
    return result


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


def save_comparison(experiments, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "mode", "topology", "learning_rate", "optimizer", "batch_size",
            "epochs", "training_accuracy", "training_mse",
            "validation_accuracy", "validation_mse", "train_seconds",
        ))
        writer.writeheader()
        for item in experiments:
            writer.writerow({
                "mode": item["mode"],
                "topology": "-".join(map(str, item["topology"])),
                "learning_rate": item["learning_rate"],
                "optimizer": item["optimizer"],
                "batch_size": item["batch_size"],
                "epochs": item["epochs"],
                "training_accuracy": item["training"]["accuracy"],
                "training_mse": item["training"]["mse"],
                "validation_accuracy": item["validation"]["accuracy"],
                "validation_mse": item["validation"]["mse"],
                "train_seconds": item["train_seconds"],
            })


def run(train_path, test_path, output_dir, seed, epochs, mini_batch_size,
        validation_fraction):
    if epochs < 1 or mini_batch_size < 2:
        raise ValueError("epochs must be positive and mini batch size at least 2")
    images, labels = read_digits(train_path)
    train_indices, val_indices = stratified_split(labels, validation_fraction, seed)
    train_images, train_labels = images[train_indices], labels[train_indices]
    train_data = list(zip(train_images, targets(train_labels)))
    val_images, val_labels = images[val_indices], labels[val_indices]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare the required architecture, learning rate, and optimizer variants.
    search = []
    for hidden in ARCHITECTURES:
        for rate in LEARNING_RATES:
            for optimizer_name in OPTIMIZERS:
                topology = (784, *hidden, 10)
                search.append(run_configuration(
                    train_data, train_images, train_labels, val_images, val_labels,
                    topology, rate,
                    optimizer_name, mini_batch_size, epochs, seed, "mini",
                ))
    key = lambda item: (item["validation"]["accuracy"],
                        -item["validation"]["mse"])
    best_mini = max(search, key=key)

    # Hold the chosen hyperparameters fixed to compare update frequencies.
    mode_comparison = [best_mini]
    for mode, batch_size in (("online", 1), ("batch", -1)):
        mode_comparison.append(run_configuration(
            train_data, train_images, train_labels, val_images, val_labels,
            best_mini["topology"],
            best_mini["learning_rate"], best_mini["optimizer"],
            batch_size, epochs, seed, mode,
        ))
    selected = max(mode_comparison, key=key)
    experiments = search + mode_comparison[1:]

    # Refit the selected configuration on every example from digits.csv.
    final_model = make_model(selected["topology"], selected["learning_rate"],
                             selected["optimizer"], seed)
    final_data = list(zip(images, targets(labels)))
    final_model.train(final_data, epochs=epochs, epsilon=-1,
                      batch_size=selected["batch_size"], seed=seed)
    model_path = output_dir / "digit_model.model"
    final_model.save_path = model_path
    final_model.save()

    # The test file is first opened after all experiment choices are fixed.
    test_images, test_labels = read_digits(test_path)
    predictions = final_model.classify(test_images).argmax(axis=1)
    result = {
        "seed": seed, "epochs": epochs, "mini_batch_size": mini_batch_size,
        "validation_fraction": validation_fraction,
        "train_count": len(train_indices), "validation_count": len(val_indices),
        "learning_count": len(labels), "test_count": len(test_labels),
        "learning_class_counts": np.bincount(labels, minlength=10).tolist(),
        "search": search, "mode_comparison": mode_comparison,
        "selected": selected,
        "test": classification_report(predictions, test_labels),
        "model_path": str(model_path),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    save_comparison(experiments, output_dir / "comparison.csv")
    print(f"Selected: {selected['mode']} {selected['topology']} "
          f"{selected['optimizer']} lr={selected['learning_rate']}")
    print(f"Test accuracy={result['test']['accuracy']:.4f}; "
          f"macro F1={result['test']['macro_f1']:.4f}")
    print(f"Results saved to {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "digits.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "digits_test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ej2_digitos")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--mini-batch-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    run(args.train, args.test, args.output_dir, args.seed, args.epochs,
        args.mini_batch_size, args.validation_fraction)


if __name__ == "__main__":
    main()
