"""Run the small validation problems from the TP3 assignment."""
import argparse
import json
from pathlib import Path

import numpy as np

from models.activation import Adaline, Step, Tanh
from models.neural_network import MultiLayerPerceptron, SimplePerceptron


def evaluate(model, inputs, targets, classification):
    predictions = np.array([model.classify(x).item() for x in inputs])

    metrics = {"mse": float(np.mean((predictions - targets) ** 2))}
    if classification:
        classes = np.where(predictions >= 0, 1, -1)
        metrics["accuracy"] = float(np.mean(classes == targets))
    return predictions, metrics


def run_case(name, model, inputs, targets, max_epochs, check_every, mse_limit):
    classification = name in {"and", "xor"}
    if isinstance(model, SimplePerceptron):
        training_data = list(zip(inputs, targets))
    else:
        training_data = [(x, np.array([target])) for x, target in zip(inputs, targets)]

    history = []
    for epoch in range(max_epochs + 1):
        if epoch > 0:
            # A negative epsilon forces one complete epoch in the model's train method.
            model.train(training_data, epochs=1, epsilon=-1)
        if epoch % check_every != 0 and epoch != max_epochs:
            continue

        predictions, metrics = evaluate(model, inputs, targets, classification)
        history.append({"epoch": epoch, **metrics})
        passed = metrics["mse"] <= mse_limit
        if classification:
            passed = passed and metrics["accuracy"] == 1.0
        if passed:
            break

    return {
        "passed": passed, # type: ignore
        "epochs_run": epoch, # type: ignore
        "max_epochs": max_epochs,
        "mse_limit": mse_limit,
        "inputs": inputs.tolist(),
        "targets": targets.tolist(),
        "predictions": predictions.tolist(), # type: ignore
        "metrics": metrics, # type: ignore
        "history": history,
        "weights": model.weights.tolist(),
    }


def run_validation(seed):
    logic_inputs = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)
    line_inputs = np.linspace(-1, 1, 50).reshape(-1, 1)
    cases = {}

    np.random.seed(seed)
    cases["and"] = run_case(
        "and", SimplePerceptron(2, Step(), 0.1, (-1, 1)), logic_inputs,
        np.array([-1, -1, -1, 1], dtype=float), 100, 1, 0.0,
    )
    cases["and"]["model"] = {"activation": "step", "learning_rate": 0.1}

    np.random.seed(seed)
    cases["linear"] = run_case(
        "linear", SimplePerceptron(1, Adaline(), 0.05, (-1, 1)), line_inputs,
        line_inputs[:, 0].copy(), 100, 1, 1e-4,
    )
    cases["linear"]["model"] = {"activation": "linear", "learning_rate": 0.05}

    np.random.seed(seed)
    cases["nonlinear"] = run_case(
        "nonlinear", SimplePerceptron(1, Tanh(), 0.05, (-1, 1)), line_inputs,
        np.tanh(line_inputs[:, 0]), 100, 1, 1e-4,
    )
    cases["nonlinear"]["model"] = {
        "activation": "tanh", "beta": 1.0, "learning_rate": 0.05,
    }

    np.random.seed(seed)
    topology = [2, 3, 2, 1]
    cases["xor"] = run_case(
        "xor", MultiLayerPerceptron(topology, Tanh(), 0.1), logic_inputs,
        np.array([1, 1, -1, -1], dtype=float), 5000, 25, 0.005,
    )
    cases["xor"]["model"] = {
        "activation": "tanh", "beta": 1.0, "learning_rate": 0.1,
        "topology": topology,
    }

    return {"seed": seed, "cases": cases, "passed": all(c["passed"] for c in cases.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/validation.json"))
    args = parser.parse_args()

    result = run_validation(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for name, case in result["cases"].items():
        metrics = case["metrics"]
        accuracy = f", accuracy={metrics['accuracy']:.0%}" if "accuracy" in metrics else ""
        status = "OK" if case["passed"] else "FAIL"
        print(f"{name}: {status} (epochs={case['epochs_run']}, mse={metrics['mse']:.6f}{accuracy})")
    print(f"Results: {args.output}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
