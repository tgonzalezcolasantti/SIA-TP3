"""Study digit classification with the shared multicapa perceptron."""

import argparse
import csv
import json
import multiprocessing
from multiprocessing.pool import ApplyResult, ThreadPool
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
# Small matrix products are usually faster with one BLAS worker per process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from ej2_digitos.train_model import fit_model, targets
from ej2_digitos.utils import ARCHITECTURES, LEARNING_RATES, OPTIMIZERS, ROOT, classification_report, read_digits


import numpy as np



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

def build_params(train_path, test_path, output_dir, seed, learning_rate, topology, optimizer, mode, epochs, batch_size, val_fraction) -> str:
    return f"--train {str(train_path)} --test {str(test_path)} --output-dir {str(output_dir)} --seed {seed} --learning-rate {learning_rate:.6g}"+\
           f" --topology {str(list(topology)).replace(" ", "")} --optimizer {optimizer} --mode {mode} --epochs {epochs} --batch-size {batch_size} --validation-fraction {val_fraction:.6g}"

def run_task(params: str, task: TaskID, progress: Progress) -> Dict[str, Any]:
    cmd: List[str] = ["uv", "run", "python", "-m", "ej2_digitos.train_model", *params.split()]
    if progress:
        progress.start_task(task)
        progress.update(task, visible=True)
    gen = 0
    with subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1) as proc:
        while True:
            result = proc.poll()
            if result is not None:
                break
            if proc.stdout:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if "Epoch" in line and progress is not None:
                        gen = int(line.split()[1])
                        progress.update(task, completed=gen, refresh=True)
                    elif "{" in line:
                        progress.remove_task(task)
                        return json.loads(line)
                    time.sleep(0)

        if progress:
            progress.remove_task(task)
        if result != 0:
            print(f"  ERROR: {params}")
            if proc.stderr is not None:
                print(proc.stderr.readlines())
        return {}

def print_row(row: Dict[str, object]) -> None:
    print(f"RESULT: {row['mode']:6} {str(row['topology']):20} {row['optimizer']:8} lr={row['learning_rate']}: "
    f"accuracy={row['validation']['accuracy']:.4f} ({row['train_seconds']:.1f}s)") # type: ignore


def run(train_path, test_path, output_dir, seed, epochs, mini_batch_size,
        validation_fraction, tasks):
    if epochs < 1 or mini_batch_size < 2:
        raise ValueError("epochs must be positive and mini batch size at least 2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare the required architecture, learning rate, and optimizer variants.
    jobs = []
    taskprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )
    globalprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    table = Table(box=None)
    table.add_row(taskprogress)
    table.add_row(globalprogress)
    rows = []
    with (
        ThreadPool(processes=tasks or int(multiprocessing.cpu_count())) as executor,
        Live(table, refresh_per_second=10),
    ):
        jobs: List[ApplyResult] = []
        for hidden in ARCHITECTURES:
            for rate in LEARNING_RATES:
                for optimizer_name in OPTIMIZERS:
                    topology = (784, *hidden, 10)
                    task = taskprogress.add_task(
                        f"{"mini":6} {topology} {optimizer_name:8} lr={rate}",
                        start=False,
                        total=epochs,
                        visible=False,
                        is_task=True,
                    )
                    params = build_params(train_path, test_path, output_dir, seed, rate, topology, optimizer_name, "mini", epochs, mini_batch_size, validation_fraction)
                    jobs.append(
                        executor.apply_async(
                            run_task, (params, task, taskprogress)
                        )
                    )
        full_progress = globalprogress.add_task(
            f"Total progress ({len(jobs)} elements)", total=len(jobs), is_task=False
        )
        with open(output_dir / "metrics.csv", mode="w+") as csv_file:
            writer = None
            while len(jobs) > 0:
                for job in list(jobs):
                    if job.ready():
                        jobs.remove(job)
                        row: Dict[str, Any] = job.get()
                        globalprogress.advance(full_progress)
                        rows.append(row)
                        print_row(row)
                        csv_stuff = {'mode': row['mode'], 'topology': row['topology'], 'learning_rate': row['learning_rate'],
                                        'optimizer': row['optimizer'], 'training_accuracy': row['training']['accuracy'], 'validation_accuracy': row['validation']['accuracy'] }
                        if writer is None:
                            writer = csv.DictWriter(csv_file, csv_stuff.keys())
                            writer.writeheader()
                        writer.writerow(csv_stuff)
        key = lambda item: (item["validation"]["accuracy"],
                            -item["validation"]["mse"])
        best_mini = max(rows, key=key)

        # Hold the chosen hyperparameters fixed to compare update frequencies.
        mode_comparison = [best_mini]
        for mode, batch_size in (("online", 1), ("batch", -1)):
            task = taskprogress.add_task(
                f"{mode:6} {best_mini["topology"]} {best_mini["optimizer"]:8} lr={best_mini["learning_rate"]}",
                start=False,
                total=epochs,
                visible=False,
                is_task=True,
            )
            params = build_params(train_path, test_path, output_dir, seed, best_mini["learning_rate"], best_mini["topology"], best_mini["optimizer"], mode, epochs, batch_size, validation_fraction)
            jobs.append(
                executor.apply_async(
                    run_task, (params, task, taskprogress)
                )
            )
        while len(jobs) > 0:
            for job in list(jobs):
                if job.ready():
                    jobs.remove(job)
                    row: Dict[str, Any] = job.get()
                    mode_comparison.append(row)
                    print_row(row)
        selected = max(mode_comparison, key=key)
        experiments = rows + mode_comparison[1:]

    # # Refit the selected configuration on every example from digits.csv.
        images, labels = read_digits(train_path)
        final_data = list(zip(images, targets(labels)))
        final_model, _ = fit_model(final_data, selected["topology"],
                                selected["learning_rate"], selected["optimizer"],
                                selected["batch_size"], epochs, seed, output_dir, label="final")
        model_path = output_dir / "digit_model.model"
        final_model.save_path = model_path
        final_model.save()

    # The test file is first opened after all experiment choices are fixed.
    test_images, test_labels = read_digits(test_path)
    predictions = final_model.classify(test_images).argmax(axis=1)
    result = {
        "seed": seed, "epochs": epochs, "mini_batch_size": mini_batch_size,
        "validation_fraction": validation_fraction,
        "learning_count": len(labels), "test_count": len(test_labels),
        "learning_class_counts": np.bincount(labels, minlength=10).tolist(),
        "search": rows, "mode_comparison": mode_comparison,
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--mini-batch-size", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    run(args.train, args.test, args.output_dir, args.seed, args.epochs,
        args.mini_batch_size, args.validation_fraction, args.max_tasks)


if __name__ == "__main__":
    main()
