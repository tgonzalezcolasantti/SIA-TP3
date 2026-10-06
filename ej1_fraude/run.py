"""Compare simple perceptrons and evaluate fraud probability distillation."""

import argparse
import csv
import io
import json
from pathlib import Path

import matplotlib
import numpy as np

import matplotlib.pyplot as plt

from models.neural_network import SimplePerceptron
from models.activation import Adaline, Logistic

matplotlib.use("Agg")

TARGET = "big_model_fraud_probability"
GROUND_TRUTH = "flagged_fraud"
EPOCHS = 100
LEARNING_RATE = 0.001
BETA = 1.0
FOLDS = 5
CHECKPOINTS = {0, 1, 2, 5, 10, 20, 50, 100}


def load_data(dataset_path):
    raw = dataset_path.read_bytes()

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    feature_names = [name for name in reader.fieldnames if name not in {TARGET, GROUND_TRUTH}] # type: ignore
    rows = list(reader)
    if not rows:
        raise ValueError("fraud_dataset.csv has no data rows")
    if any(value is None or value.strip() == "" for row in rows for value in row.values()):
        raise ValueError("fraud_dataset.csv contains missing values")

    inputs = np.array([[float(row[name]) for name in feature_names] for row in rows])
    teacher = np.array([float(row[TARGET]) for row in rows])
    flags = np.array([int(row[GROUND_TRUTH]) for row in rows])
    if not np.isfinite(inputs).all() or not np.isfinite(teacher).all():
        raise ValueError("fraud_dataset.csv contains non-finite values")
    if np.any((teacher < 0) | (teacher > 1)) or not set(flags).issubset({0, 1}):
        raise ValueError("target columns contain values outside their documented ranges")
    return inputs, teacher, flags, feature_names


def describe_data(inputs, teacher, flags, feature_names):
    columns = {}
    for index, name in enumerate(feature_names):
        values = inputs[:, index]
        columns[name] = {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
            "unique": int(len(np.unique(values))),
        }
    return {
        "rows": len(inputs),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "duplicate_rows": int(len(inputs) - len(np.unique(
            np.column_stack((inputs, teacher, flags)), axis=0
        ))),
        "fraud_count": int(flags.sum()),
        "fraud_rate": float(flags.mean()),
        "teacher_probability": {
            "min": float(teacher.min()),
            "median": float(np.median(teacher)),
            "max": float(teacher.max()),
        },
        "columns": columns,
    }


def fit_scaler(inputs):
    means = inputs.mean(axis=0)
    stds = inputs.std(axis=0)
    stds[stds == 0] = 1
    return means, stds


def make_model(kind, features, seed, name, savedir):
    np.random.seed(seed)
    activation = Adaline() if kind == "linear" else Logistic(BETA)
    return SimplePerceptron(
        inputs=features,
        activation=activation,
        learning_rate=LEARNING_RATE,
        save=savedir, 
        model_name=name,
        weight_range=(-0.05, 0.05)
    )


def regression_metrics(predictions, targets):
    errors = predictions - targets
    return {
        "mse": float(np.mean(errors ** 2)),
        "mae": float(np.mean(np.abs(errors))),
    }


def train(model, inputs, teacher, epochs, checkpoints=None, with_history: bool = False):
    training_data = list(zip(inputs, teacher))
    history = []
    # Cutoff on full epochs so that post-update learning curves are comparable.
    errors = model.train(training_data, epochs=epochs, epsilon=-1, with_history=with_history)
    if with_history:
        for epoch in range(epochs):
            if checkpoints is None or epoch in checkpoints:
                try:
                    history.append({"epoch": epoch, "mse": errors[epoch][0], "mae": errors[epoch][1]})
                except IndexError:
                    continue #oops
    return history


def compare_learning(inputs, teacher, seed, savedir):
    means, stds = fit_scaler(inputs)
    scaled = (inputs - means) / stds
    comparison = {}
    for kind in ("linear", "logistic"):
        model = make_model(kind, scaled.shape[1], seed, "Compare", savedir)
        history = train(model, scaled, teacher, EPOCHS, CHECKPOINTS, with_history=True)
        predictions = model.classify(scaled).flatten()
        comparison[kind] = {
            "history": history,
            "final": regression_metrics(predictions, teacher),
            "prediction_range": [float(predictions.min()), float(predictions.max())],
            "outside_probability_range": int(np.sum((predictions < 0) | (predictions > 1))),
            "weights": model.weights.tolist(),
        }
    return comparison


def cross_validate(inputs, teacher, flags, seed, savedir):
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(inputs)), FOLDS)
    oof = np.empty(len(inputs))
    nested_decisions = np.empty(len(inputs), dtype=bool)
    folds = []
    for fold_index, held_out in enumerate(parts):
        training = np.concatenate([part for i, part in enumerate(parts) if i != fold_index])

        # Choose the threshold using only data inside this outer training fold.
        inner = np.random.default_rng(seed + 100 + fold_index).permutation(training)
        inner_train, calibration = np.split(inner, [int(0.8 * len(inner))])
        inner_means, inner_stds = fit_scaler(inputs[inner_train])
        inner_model = make_model("logistic", inputs.shape[1], seed + 100 + fold_index, f"Crossval_{fold_index}", savedir)
        train(inner_model, (inputs[inner_train] - inner_means) / inner_stds,
              teacher[inner_train], EPOCHS)
        calibration_predictions = inner_model.classify((inputs[calibration] - inner_means) / inner_stds).flatten()
        inner_threshold = choose_threshold(calibration_predictions, flags[calibration])["threshold"]

        means, stds = fit_scaler(inputs[training])
        scaled_train = (inputs[training] - means) / stds
        scaled_test = (inputs[held_out] - means) / stds
        model = make_model("logistic", inputs.shape[1], seed + fold_index, "Crossval_final", savedir)
        train(model, scaled_train, teacher[training], EPOCHS)
        predictions =model.classify(scaled_test).flatten()
        oof[held_out] = predictions
        nested_decisions[held_out] = predictions >= inner_threshold
        folds.append({
            "fold": fold_index + 1,
            "train_count": len(training),
            "validation_count": len(held_out),
            "validation_fraud_count": int(flags[held_out].sum()),
            "inner_threshold": inner_threshold,
            "outer_classification": classification_metrics(
                predictions, flags[held_out], inner_threshold
            ),
            **regression_metrics(predictions, teacher[held_out]),
        })
        print(f"Fold {fold_index + 1}/{FOLDS}: MSE={folds[-1]['mse']:.6f}")
    nested_metrics = classification_metrics(nested_decisions.astype(float), flags, 0.5)
    del nested_metrics["threshold"]
    return oof, folds, nested_metrics


def classification_metrics(probabilities, flags, threshold):
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (flags == 1)))
    fp = int(np.sum(predicted & (flags == 0)))
    fn = int(np.sum(~predicted & (flags == 1)))
    tn = int(np.sum(~predicted & (flags == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if tp + fn + fp else 0.0
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "f2": f2,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def choose_threshold(probabilities, flags):
    order = np.argsort(-probabilities, kind="stable")
    print(probabilities, flush=True)
    sorted_probabilities = probabilities[order]
    sorted_flags = flags[order]
    tp = np.cumsum(sorted_flags)
    fp = np.arange(1, len(flags) + 1) - tp
    fn = int(flags.sum()) - tp
    f2 = 5 * tp / (5 * tp + 4 * fn + fp)
    # Evaluate each distinct threshold, preferring the higher one on ties.
    last_at_value = np.r_[sorted_probabilities[1:] != sorted_probabilities[:-1], True]
    candidates = np.flatnonzero(last_at_value)
    chosen = candidates[np.argmax(f2[candidates])]
    return classification_metrics(probabilities, flags, sorted_probabilities[chosen])


def save_plots(comparison, teacher, oof, flags, threshold, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, result in comparison.items():
        history = result["history"]
        ax.plot([point["epoch"] for point in history],
                [point["mse"] for point in history], marker="o", label=name)
    ax.set(xlabel="Época", ylabel="MSE de entrenamiento", title="Aprendizaje con 7.500 muestras")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(teacher, oof, c=flags, cmap="coolwarm", s=5, alpha=0.35)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.axhline(threshold, color="tab:red", linestyle=":", label=f"Umbral {threshold:.3f}")
    ax.set(xlabel="Probabilidad de BigModel", ylabel="Predicción fuera de muestra de TinyModel",
           title="Generalización: predicciones de 5 folds", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "oof_predictions.png", dpi=160)
    plt.close(fig)


def run(dataset_path, output_dir, seed):
    inputs, teacher, flags, feature_names = load_data(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = describe_data(inputs, teacher, flags, feature_names)

    print("Comparing linear and logistic learning on all samples...")
    comparison = compare_learning(inputs, teacher, seed, output_dir)
    print("Evaluating logistic generalization with 5 folds...")
    oof, folds, nested_threshold_metrics = cross_validate(inputs, teacher, flags, seed, None)
    threshold_metrics = choose_threshold(oof, flags)
    oof_metrics = regression_metrics(oof, teacher)

    means, stds = fit_scaler(inputs)
    final_model = make_model("logistic", inputs.shape[1], seed, None, output_dir / "tiny.model")
    train(final_model, (inputs - means) / stds, teacher, EPOCHS)
    threshold = threshold_metrics["threshold"]
    np.savez(output_dir / "tiny_model.npz", weights=final_model.weights,
             means=means, stds=stds, feature_names=np.array(feature_names),
             beta=BETA, threshold=threshold)
    save_plots(comparison, teacher, oof, flags, threshold, output_dir)

    result = {
        "source_dataset": str(dataset_path),
        "seed": seed,
        "configuration": {
            "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "beta": BETA,
            "folds": FOLDS, "scaling": "mean/std fitted on each training fold",
            "threshold_rule": "maximize F2 on out-of-fold predictions",
        },
        "data_profile": profile,
        "learning_comparison_all_samples": comparison,
        "generalization": {
            "folds": folds,
            "mean_fold_mse": float(np.mean([fold["mse"] for fold in folds])),
            "std_fold_mse": float(np.std([fold["mse"] for fold in folds], ddof=1)),
            "out_of_fold": oof_metrics,
            "nested_threshold_evaluation": nested_threshold_metrics,
            "threshold_selection": threshold_metrics,
            "threshold_at_0_5": classification_metrics(oof, flags, 0.5),
        },
        "final_model": {
            "learnable_parameters": len(final_model.weights),
            "weights": final_model.weights.tolist(),
            "threshold": threshold,
            "path": str(output_dir / "tiny_model.npz"),
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected threshold: {threshold:.6f}; OOF MSE: {oof_metrics['mse']:.6f}")
    print(f"Results saved in {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "fraud_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ej1_fraude"))
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    run(args.dataset, args.output_dir, args.seed)


if __name__ == "__main__":
    import cProfile
    # cProfile.run('main()')
    main()
