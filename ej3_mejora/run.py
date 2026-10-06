"""Improve digit classification using the additional examples from exercise 3."""

import argparse
import csv
import json
import multiprocessing
from multiprocessing.pool import ApplyResult, ThreadPool
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List

# Small matrix products are usually faster with one BLAS worker per process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from ej2_digitos.utils import ROOT, fit_model, read_digits, classification_report
from ej3_mejora.train_model import training_indices, make_training_data, load_learning_data, stratified_split


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

    search = sorted(search, key=lambda x: x['validation']['accuracy'])

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

def print_row(result: Dict[str, Any]):
    print(f"{result['source']:8} {result['topology']} lr={result['learning_rate']} "
          f"balance={result['balance_count']} shifts={result['shifts_per_image']}: validation="
          f"{result['validation']['accuracy']:.2%}, "
          f"recall 8={result['validation']['recall_8']:.2%} ({result['train_seconds']:.1f}s)", flush=True)

def run_task(params: Dict, old_path: Path, new_path: Path, output_path: Path, validation_fraction: float, seed: int, i: int, task: TaskID, progress: Progress) -> Dict[str, Any]:
    cmd: List[str] = ["uv", "run", "python", "-m", "ej3_mejora.train_model", "--old-path", str(old_path), "--new-path", str(new_path), 
                      "--validation-fraction", f"{validation_fraction:.6g}", "--seed", str(seed), "--source", params['source'],
                      "--topology", str(params['topology']).replace(" ", ""), "--learning-rate", str(params['learning_rate']),
                      "--balance-count", str(params['balance_count']), "--shifts-per-image", str(params['shifts_per_image']),
                      "--epochs", str(params['epochs']), "--batch-size", str(params['batch_size']), "--output-path", str(output_path), "--label", str(i)]
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
                        line = line.replace("\'", "\"")
                        print(line, flush=True)
                        return json.loads(line)
                    time.sleep(0)

        if progress:
            progress.remove_task(task)
        if result != 0:
            print(f"  ERROR: {params}", flush=True)
            if proc.stderr is not None:
                print(proc.stderr.readlines(), flush=True)
        return {}

def run(old_path, new_path, test_path, output_dir, seed=2, epochs=40,
        batch_size=128, balance_count=2000, validation_fraction=0.2, tasks = None):
    if epochs < 1 or batch_size < 1 or balance_count < 0:
        raise ValueError("epochs and batch size must be positive; balance count nonnegative")
    images, labels, from_new, profile = load_learning_data(old_path, new_path)
    train, validation = stratified_split(labels, validation_fraction, seed)
    new_train = train[from_new[train]]
    pools = {"new": new_train, "combined": train}
    output_dir.mkdir(parents=True, exist_ok=True)
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
    search = []
    with (
        ThreadPool(processes=tasks or int(multiprocessing.cpu_count())) as executor,
        Live(table, refresh_per_second=10),
    ):
        jobs: List[ApplyResult] = []
        for i, config in enumerate(configurations(epochs, batch_size, balance_count)):
            task = taskprogress.add_task(
                str(config),
                start=False,
                total=epochs,
                visible=False,
                is_task=True,
            )
            jobs.append(
                executor.apply_async(
                    run_task, (config, old_path, new_path, output_dir, validation_fraction, seed, i, task, taskprogress)
                )
            )
        full_progress = globalprogress.add_task(
            f"Total progress ({len(jobs)} elements)", total=len(jobs), is_task=False
        )
        while len(jobs) > 0:
            for job in list(jobs):
                if job.ready():
                    jobs.remove(job)
                    row: Dict[str, Any] = job.get()
                    globalprogress.advance(full_progress)
                    search.append(row)
                    print_row(row)
    selected = max(search, key=lambda result: (
        result["validation"]["accuracy"], -result["validation"]["mse"]
    ))

    # Refit on all eligible learning images after selecting with validation.
    final_pool = np.flatnonzero(from_new) if selected["source"] == "new" else np.arange(len(labels))
    final_indices = training_indices(final_pool, labels, selected["balance_count"], seed)
    final_data = make_training_data(images, labels, final_indices,
                                    selected["shifts_per_image"], seed)
    final_model, _ = fit_model(final_data, selected["topology"],
                               selected["learning_rate"], "RMSProp",
                               selected["batch_size"], selected["epochs"], seed, output_dir, "final")

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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--balance-count", type=int, default=2000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    run(args.old, args.new, args.test, args.output_dir, args.seed, args.epochs,
        args.batch_size, args.balance_count, args.validation_fraction, 4)


if __name__ == "__main__":
    main()
