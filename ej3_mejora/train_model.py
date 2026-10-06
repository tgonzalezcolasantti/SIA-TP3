
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ej2_digitos.utils import ROOT, fit_model, evaluate, stratified_split, targets

MINORITY_DIGITS = (5, 8)
SHIFTS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
          (0, 1), (1, -1), (1, 0), (1, 1))

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

def make_training_data(images, labels, indices, shifts_per_image, seed):
    selected_images = shift_images(images[indices], shifts_per_image, seed)
    selected_labels = np.tile(labels[indices], shifts_per_image + 1)
    return list(zip(selected_images, targets(selected_labels)))

def fit_candidate(config, pool, images, labels, validation, seed, output_path, label):
    train_indices = training_indices(pool, labels, config["balance_count"], seed)
    training_data = make_training_data(images, labels, train_indices,
                                       config["shifts_per_image"], seed)
    model, seconds = fit_model(training_data, config["topology"],
                               config["learning_rate"], "RMSProp",
                               config["batch_size"], config["epochs"], seed, output_path, label)
    result = {
        **config, "optimizer": "momentum", "train_unique_count": len(pool),
        "train_draw_count": len(train_indices),
        "train_augmented_count": len(training_data), "train_seconds": seconds,
        "training": evaluate(model, images[pool], labels[pool], (5, 8)),
        "validation": evaluate(model, images[validation], labels[validation], (5, 8)),
    }
    result['topology'] = [int(x) for x in result['topology']]
    print(result)
    print(json.dumps(result), flush=True)
    return result

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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-path", type=Path)
    parser.add_argument("--new-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--source", type=str)
    parser.add_argument("--topology", type=str)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--balance-count", type=int)
    parser.add_argument("--shifts-per-image", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    images, labels, from_new, _ = load_learning_data(args.old_path, args.new_path)
    train, validation = stratified_split(labels, args.validation_fraction, args.seed)
    new_train = train[from_new[train]]
    pools = {"new": new_train, "combined": train}
    config = {"source": args.source, "topology": np.fromstring(args.topology[1:-1], sep=",", dtype=np.int32),
              "learning_rate": args.learning_rate, "balance_count": args.balance_count,
              "shifts_per_image": args.shifts_per_image, "epochs": args.epochs, "batch_size": args.batch_size}
    fit_candidate(config, pools[config["source"]], images, labels, validation, args.seed, args.output_path, args.label)

if __name__ == '__main__':
    main()