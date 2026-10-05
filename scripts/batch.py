"""Run independent exercise 1 experiments concurrently."""

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "seed", "status", "oof_mse", "oof_mae", "threshold", "nested_f2",
    "result_dir", "error",
)


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    return seeds


def run_one(seed: int, dataset: Path, output_dir: Path, timeout: float) -> dict:
    result_dir = output_dir / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "ej1_fraude.run", "--dataset", str(dataset),
        "--output-dir", str(result_dir), "--seed", str(seed),
    ]
    row = dict.fromkeys(FIELDS, "")
    row.update(seed=seed, status="failed", result_dir=str(result_dir))
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        (result_dir / "run.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            row["error"] = f"process exited with code {completed.returncode}"
            return row
        result = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
        generalization = result["generalization"]
        row.update(
            status="ok",
            oof_mse=generalization["out_of_fold"]["mse"],
            oof_mae=generalization["out_of_fold"]["mae"],
            threshold=generalization["threshold_selection"]["threshold"],
            nested_f2=generalization["nested_threshold_evaluation"]["f2"],
        )
    except subprocess.TimeoutExpired as exc:
        row["error"] = f"timed out after {exc.timeout} seconds"
    except (OSError, ValueError, KeyError) as exc:
        row["error"] = str(exc)
    return row


def run_batch(seeds: list[int], dataset: Path, output_dir: Path,
              tasks: int, timeout: float) -> list[dict]:
    if tasks < 1 or timeout <= 0:
        raise ValueError("tasks and timeout must be positive")
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=tasks) as executor:
        futures = {
            executor.submit(run_one, seed, dataset, output_dir, timeout): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"Seed {row['seed']}: {row['status']}", flush=True)
    rows.sort(key=lambda row: seeds.index(row["seed"]))
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_seeds, default=[2, 3, 4])
    parser.add_argument("--tasks", type=int, default=2,
                        help="Maximum number of simultaneous runs")
    parser.add_argument("--timeout", type=float, default=1200,
                        help="Time limit per run, in seconds")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "fraud_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "batch_ej1")
    args = parser.parse_args()
    rows = run_batch(args.seeds, args.dataset.resolve(), args.output_dir.resolve(),
                     args.tasks, args.timeout)
    print(f"Summary saved to {args.output_dir / 'summary.csv'}")
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
