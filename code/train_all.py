"""
Train every (case, model, problem) combination with the best hyperparameters
found by the hyperparameter search (run_all.py -> hparam_search.py).

For each combination the best config is loaded from
    ray_results/best_config_{model}_{problem}_{mode}_{case_name}.json
and main.py is run with it; test metrics are written to
    train_results/results_{model}_{problem}_{mode}_{case_name}.json
"""

import json
import os
import subprocess
import sys
from itertools import product

# Same combinations as run_all.py
MODELS = ["dnn", "e2elr", "e2elrdc"]
PROBLEMS = ["edr"]
CASES = ["pglib_opf_case300_ieee.m", "pglib_opf_case1354_pegase.m"]
MODE = "ssl"
RESULTS_DIR = "ray_results"
TRAIN_RESULTS_DIR = "train_results"


def load_best_config(model: str, problem: str, case_name: str) -> dict | None:
    """Best config saved by hparam_search.py, or None if it does not exist."""
    path = os.path.join(
        RESULTS_DIR, f"best_config_{model}_{problem}_{MODE}_{case_name}.json"
    )
    if not os.path.isfile(path):
        print(f"  [SKIP] Config file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def get_command(
    model: str, problem: str, case_path: str, config: dict, results_file: str
) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--model", model,
        "--problem", problem,
        "--case", case_path,
        "--mode", MODE,
        "--n_layers", str(config["n_layers"]),
        "--hidden_dim", str(config["hidden_dim"]),
        "--lr", str(config["lr"]),
        "--batch_size", str(config["batch_size"]),
        "--skip_solve",
        "--results_file", results_file,
    ]  # fmt: skip


if __name__ == "__main__":
    failed = []
    for case_file, model, problem in product(CASES, MODELS, PROBLEMS):
        case_name = os.path.splitext(case_file)[0]
        print(f"\n{'=' * 60}")
        print(f"case={case_file}  model={model}  problem={problem}")
        print(f"{'=' * 60}")

        loaded = load_best_config(model, problem, case_name)
        if loaded is None:
            continue
        print(f"Best val_loss from search: {loaded['best_val_loss']:.6f}")
        print(f"Best config: {loaded['config']}")

        results_file = os.path.join(
            TRAIN_RESULTS_DIR, f"results_{model}_{problem}_{MODE}_{case_name}.json"
        )
        command = get_command(
            model,
            problem,
            os.path.join("data", case_file),
            loaded["config"],
            results_file,
        )
        print(f"Command: {' '.join(command)}")
        if subprocess.run(command).returncode != 0:
            failed.append((case_file, model, problem))

    print(f"\nFailed runs: {failed}" if failed else "\nAll training runs complete.")
