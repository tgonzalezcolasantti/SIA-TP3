"""Improve digit classification using the additional examples from exercise 3."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from ej2_digitos.run import (classification_report, evaluate, fit_model,
                            read_digits, stratified_split, targets)


ROOT = Path(__file__).resolve().parents[1]
MINORITY_DIGITS = (5, 8)
SHIFTS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
          (0, 1), (1, -1), (1, 0), (1, 1))


def load_learning_data(old_path: Path, new_path: Path):
    """Keep each labelled image once, preferring the new file on overlap."""
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    if set(old.columns) != {"label", "image"} or set(new.columns) != {"label", "image"}:
        raise ValueError("digit files must contain label and image columns")
    old["from_new"] = False
    new["from_new"] = True
    combined = pd.concat((new, old), ignore_index=True)
    combined = combined.drop_duplicates(subset=["label", "image"])
    images = np.stack([
        np.fromstring(value[1:-1], dtype=np.float32, sep=",")
        for value in combined["image"]
    ])
    labels = combined["label"].to_numpy(dtype=np.int64)
    if images.shape != (len(labels), 784) or not np.isfinite(images).all():
        raise ValueError("each image must contain 784 finite pixels")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("digit labels must be between 0 and 9")
    from_new = combined["from_new"].to_numpy(dtype=bool)
    return images, labels, from_new, {
        "old_rows": len(old), "new_rows": len(new),
        "shared_rows": len(old) + len(new) - len(combined),
        "unique_rows": len(combined),
        "old_class_counts": np.bincount(old["label"], minlength=10).tolist(),
        "new_class_counts": np.bincount(new["label"], minlength=10).tolist(),
        "combined_class_counts": np.bincount(labels, minlength=10).tolist(),
    }


def training_indices(pool, labels, balance_count, seed):
    """Resample minority examples only inside the training partition."""
    if balance_count == 0:
        return pool.copy()
    rng = np.random.default_rng(seed)
    pieces = [pool]
    for digit in MINORITY_DIGITS:
        members = pool[labels[pool] == digit]
        if len(members) == 0:
            raise ValueError(f"no examples of digit {digit} in training")
        if len(members) < balance_count:
            pieces.append(rng.choice(members, balance_count - len(members), replace=True))
    return np.concatenate(pieces)


def shift_images(images, shifts_per_image, seed):
    """Append small translated copies; labels remain aligned with each block."""
    if shifts_per_image == 0:
        return images
    rng = np.random.default_rng(seed)
    original = images.reshape(-1, 28, 28)
    copies = [images]
    for _ in range(shifts_per_image):
        shifted = np.zeros_like(original)
        choices = rng.integers(len(SHIFTS), size=len(images))
        for choice, (dy, dx) in enumerate(SHIFTS):
            members = np.flatnonzero(choices == choice)
            if len(members) == 0:
                continue
            source_y = slice(max(0, -dy), min(28, 28 - dy))
            source_x = slice(max(0, -dx), min(28, 28 - dx))
            target_y = slice(max(0, dy), min(28, 28 + dy))
            target_x = slice(max(0, dx), min(28, 28 + dx))
            shifted[members, target_y, target_x] = original[members, source_y, source_x]
        copies.append(shifted.reshape(-1, 784))
    return np.concatenate(copies)


def make_training_data(images, labels, indices, shifts_per_image, seed):
    selected_images = shift_images(images[indices], shifts_per_image, seed)
    selected_labels = np.tile(labels[indices], shifts_per_image + 1)
    return list(zip(selected_images, targets(selected_labels)))


def configurations(epochs, batch_size, balance_count):
    return [
        {"source": "new", "topology": [784, 128, 64, 10], "learning_rate": 0.01,
         "balance_count": 0, "shifts_per_image": 0, "epochs": epochs, "batch_size": batch_size},
        {"source": "combined", "topology": [784, 128, 64, 10], "learning_rate": 0.01,
         "balance_count": 0, "shifts_per_image": 0, "epochs": epochs, "batch_size": batch_size},
        {"source": "combined", "topology": [784, 128, 64, 10], "learning_rate": 0.01,
         "balance_count": balance_count, "shifts_per_image": 0, "epochs": epochs, "batch_size": batch_size},
        {"source": "combined", "topology": [784, 256, 128, 10], "learning_rate": 0.01,
         "balance_count": 0, "shifts_per_image": 0, "epochs": epochs, "batch_size": batch_size},
        {"source": "combined", "topology": [784, 256, 128, 10], "learning_rate": 0.01,
         "balance_count": balance_count, "shifts_per_image": 0, "epochs": epochs, "batch_size": batch_size},
        {"source": "combined", "topology": [784, 256, 128, 10], "learning_rate": 0.02,
         "balance_count": balance_count, "shifts_per_image": 0, "epochs": epochs, "batch_size": batch_size},
        {"source": "combined", "topology": [784, 256, 128, 10], "learning_rate": 0.02,
         "balance_count": balance_count, "shifts_per_image": 1, "epochs": epochs, "batch_size": batch_size},
        {"source": "combined", "topology": [784, 256, 128, 10], "learning_rate": 0.02,
         "balance_count": balance_count, "shifts_per_image": 2, "epochs": epochs, "batch_size": batch_size},
    ]


def fit_candidate(config, pool, images, labels, validation, seed):
    train_indices = training_indices(pool, labels, config["balance_count"], seed)
    training_data = make_training_data(images, labels, train_indices,
                                       config["shifts_per_image"], seed)
    model, seconds = fit_model(training_data, config["topology"],
                               config["learning_rate"], "momentum",
                               config["batch_size"], config["epochs"], seed)
    result = {
        **config, "optimizer": "momentum", "train_unique_count": len(pool),
        "train_draw_count": len(train_indices),
        "train_augmented_count": len(training_data), "train_seconds": seconds,
        "training": evaluate(model, images[pool], labels[pool], (5, 8)),
        "validation": evaluate(model, images[validation], labels[validation], (5, 8)),
    }
    print(f"{config['source']:8} {config['topology']} lr={config['learning_rate']} "
          f"balance={config['balance_count']} shifts={config['shifts_per_image']}: validation="
          f"{result['validation']['accuracy']:.2%}, "
          f"recall 8={result['validation']['recall_8']:.2%} ({seconds:.1f}s)", flush=True)
    return result


def save_comparison(search, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "source", "topology", "learning_rate", "optimizer", "batch_size",
            "epochs", "balance_count", "train_unique_count", "train_draw_count",
            "shifts_per_image", "train_augmented_count",
            "training_accuracy", "validation_accuracy", "validation_mse",
            "validation_recall_5", "validation_recall_8", "train_seconds",
        ))
        writer.writeheader()
        for result in search:
            writer.writerow({
                "source": result["source"],
                "topology": "-".join(map(str, result["topology"])),
                "learning_rate": result["learning_rate"],
                "optimizer": result["optimizer"],
                "batch_size": result["batch_size"],
                "epochs": result["epochs"],
                "balance_count": result["balance_count"],
                "shifts_per_image": result["shifts_per_image"],
                "train_unique_count": result["train_unique_count"],
                "train_draw_count": result["train_draw_count"],
                "train_augmented_count": result["train_augmented_count"],
                "training_accuracy": result["training"]["accuracy"],
                "validation_accuracy": result["validation"]["accuracy"],
                "validation_mse": result["validation"]["mse"],
                "validation_recall_5": result["validation"]["recall_5"],
                "validation_recall_8": result["validation"]["recall_8"],
                "train_seconds": result["train_seconds"],
            })


def save_plots(search, test_report, output_dir):
    import matplotlib.pyplot as plt

    names = [
        f"{'Nuevos' if item['source'] == 'new' else 'Unión'} | "
        f"{'-'.join(map(str, item['topology'][1:-1]))} | "
        f"tasa={item['learning_rate']:.2f} | "
        f"balance={item['balance_count']} | "
        f"copias={item['shifts_per_image']}"
        for item in search
    ]
    accuracies = [100 * item["validation"]["accuracy"] for item in search]
    best_index = max(range(len(search)), key=lambda index: (
        search[index]["validation"]["accuracy"],
        -search[index]["validation"]["mse"],
    ))
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(names, accuracies,
                   color=["#f58518" if i == best_index else "#4c78a8"
                          for i in range(len(search))])
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.invert_yaxis()
    ax.set(xlabel="Accuracy de validación (%)", title="Comparación de configuraciones")
    ax.set_xlim(94, 99)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_comparison.png", dpi=150)
    plt.close(fig)

    confusion = np.asarray(test_report["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set(xlabel="Dígito predicho", ylabel="Dígito real",
           title="Matriz de confusión en test",
           xticks=range(10), yticks=range(10))
    fig.colorbar(image, ax=ax, label="Imágenes")
    fig.tight_layout()
    fig.savefig(output_dir / "test_confusion.png", dpi=150)
    plt.close(fig)


def run(old_path, new_path, test_path, output_dir, seed=2, epochs=40,
        batch_size=128, balance_count=2000, validation_fraction=0.2):
    if epochs < 1 or batch_size < 1 or balance_count < 0:
        raise ValueError("epochs and batch size must be positive; balance count nonnegative")
    images, labels, from_new, profile = load_learning_data(old_path, new_path)
    train, validation = stratified_split(labels, validation_fraction, seed)
    new_train = train[from_new[train]]
    pools = {"new": new_train, "combined": train}
    output_dir.mkdir(parents=True, exist_ok=True)
    search = []
    for config in configurations(epochs, batch_size, balance_count):
        search.append(fit_candidate(config, pools[config["source"]], images,
                                    labels, validation, seed))
    selected = max(search, key=lambda result: (
        result["validation"]["accuracy"], -result["validation"]["mse"]
    ))

    # Refit on all eligible learning images after selecting with validation.
    final_pool = np.flatnonzero(from_new) if selected["source"] == "new" else np.arange(len(labels))
    final_indices = training_indices(final_pool, labels, selected["balance_count"], seed)
    final_data = make_training_data(images, labels, final_indices,
                                    selected["shifts_per_image"], seed)
    final_model, _ = fit_model(final_data, selected["topology"],
                               selected["learning_rate"], "momentum",
                               selected["batch_size"], selected["epochs"], seed)
    model_path = output_dir / "digit_model.model"
    final_model.save_path = model_path
    final_model.save()

    # digits_test.csv is opened only after every experiment choice is fixed.
    test_images, test_labels = read_digits(test_path)
    guesses = np.concatenate([
        final_model.classify(test_images[start:start + 2048]).argmax(axis=1)
        for start in range(0, len(test_labels), 2048)
    ])
    test_report = classification_report(guesses, test_labels)
    result = {
        "seed": seed, "validation_fraction": validation_fraction,
        "data": profile, "validation_count": len(validation),
        "new_train_count": len(new_train), "combined_train_count": len(train),
        "search": search, "selected": selected,
        "final_unique_count": len(final_pool), "final_draw_count": len(final_indices),
        "final_augmented_count": len(final_data),
        "test_count": len(test_labels), "test": test_report,
        "target_accuracy": 0.98, "target_reached": test_report["accuracy"] >= 0.98,
        "model_path": str(model_path),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_comparison(search, output_dir / "comparison.csv")
    save_plots(search, test_report, output_dir)
    print(f"Selected: {selected['source']} {selected['topology']} "
          f"lr={selected['learning_rate']} balance={selected['balance_count']} "
          f"shifts={selected['shifts_per_image']}")
    print(f"Test accuracy={test_report['accuracy']:.2%}; "
          f"recall 8={test_report['per_digit']['8']['recall']:.2%}; "
          f"98% target reached={result['target_reached']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=ROOT / "data" / "digits.csv")
    parser.add_argument("--new", type=Path, default=ROOT / "data" / "more_digits.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "digits_test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ej3_mejora")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--balance-count", type=int, default=2000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    run(args.old, args.new, args.test, args.output_dir, args.seed, args.epochs,
        args.batch_size, args.balance_count, args.validation_fraction)


if __name__ == "__main__":
    main()
