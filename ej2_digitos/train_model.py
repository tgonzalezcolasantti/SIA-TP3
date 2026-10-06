import argparse
import json
from pathlib import Path
import time

import numpy as np

from ej2_digitos.utils import ROOT, evaluate, fit_model, read_digits, stratified_split, targets
from models.activation import Tanh
from models.neural_network import MultiLayerPerceptron
from models.optimization import Optimization


def run_configuration(
    training_data,
    training_images,
    training_labels,
    validation_images,
    validation_labels,
    topology,
    learning_rate,
    optimizer_name,
    batch_size,
    epochs,
    seed,
    mode,
    output_dir,
):
    model, seconds = fit_model(
        training_data,
        topology,
        learning_rate,
        optimizer_name,
        batch_size,
        epochs,
        seed,
        output_dir,
        mode
    )
    train_metrics = evaluate(model, training_images, training_labels)
    metrics = evaluate(model, validation_images, validation_labels)
    result = {
        "mode": mode,
        "topology": [int(x) for x in topology],
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "training": train_metrics,
        "validation": metrics,
        "train_seconds": seconds,
    }
    print(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "digits.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "digits_test.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "ej2_digitos"
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--topology", type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    images, labels = read_digits(args.train)
    train_indices, val_indices = stratified_split(
        labels, args.validation_fraction, args.seed
    )
    train_images, train_labels = images[train_indices], labels[train_indices]
    train_data = list(zip(train_images, targets(train_labels)))
    val_images, val_labels = images[val_indices], labels[val_indices]

    result = run_configuration(
        train_data,
        train_images,
        train_labels,
        val_images,
        val_labels,
        np.fromstring(args.topology[1:-1], sep=",", dtype=np.int32),  # type: ignore
        args.learning_rate,
        args.optimizer,
        args.batch_size,
        args.epochs,
        args.seed,
        args.mode,
        args.output_dir
    )
    return result

if __name__ == "__main__":
    main()
