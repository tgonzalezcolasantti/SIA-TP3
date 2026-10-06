from pathlib import Path

from data.digit_dataset_loader import load_dataset

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ((64,), (128, 64))
LEARNING_RATES = (0.1, 0.05, 0.01, 0.001, 0.0001)
OPTIMIZERS = ("None", "Momentum", "RMSProp", "Adam")

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