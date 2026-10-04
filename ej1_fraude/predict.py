"""Load the saved TinyModel and predict fraud probabilities for transaction rows."""

from pathlib import Path

import numpy as np


def predict_rows(rows, model_path: str | Path = "results/ej1_fraude/tiny_model.npz"):
    """Return (probabilities, fraud_flags) for mappings with the documented features."""
    with np.load(model_path, allow_pickle=False) as model:
        features = model["feature_names"].tolist()
        inputs = np.array(
            [[float(row[name]) for name in features] for row in rows], dtype=float
        )
        scaled = (inputs - model["means"]) / model["stds"]
        z = scaled @ model["weights"][1:] + model["weights"][0]
        beta = float(model["beta"])
        probabilities = 1 / (1 + np.exp(np.clip(-2 * beta * z, -700, 700)))
        return probabilities, probabilities >= float(model["threshold"])
