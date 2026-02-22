"""
Train all (case, model, problem_type) combinations using the best hyperparameters found
during the hyperparameter search (run_all.py → hparam_search.py).

Strategy
--------
- Iterates over product(cases, models, problem_types) — matching run_all.py's search combos.
- For each combo, loads the best config from:
    ray_results/best_config_{model}_{problem}_{mode}_{case_name}.json
- Invokes main.py with those hyperparameters as CLI arguments via os.system.
"""

import json
import os
import sys
from itertools import product
from typing import Optional

MODE = "ssl"
RESULTS_DIR = "ray_results"
TRAIN_RESULTS_DIR = "train_results"


def load_best_config(
    model_name: str, problem_type: str, case_name: str
) -> Optional[dict]:
    """
    Load the best hyperparameter config saved by hparam_search.py.

    Returns the parsed JSON dict, or None if the file does not exist.
    """
    filename = f"best_config_{model_name}_{problem_type}_{MODE}_{case_name}.json"
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(path):
        print(f"  [SKIP] Config file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def get_command(
    model_name: str, problem_type: str, case_path: str, config: dict, results_file: str
) -> str:
    return (
        f"{sys.executable} main.py"
        f" --model {model_name}"
        f" --problem {problem_type}"
        f" --case {case_path}"
        f" --mode {MODE}"
        f" --n_layers {config['n_layers']}"
        f" --hidden_dim {config['hidden_dim']}"
        f" --lr {config['lr']}"
        f" --batch_size {config['batch_size']}"
        f" --skip_solve"
        f" --results_file {results_file}"
    )


if __name__ == "__main__":
    models = ["dnn", "e2elr", "e2elrdc"]
    problem_types = ["edr"]
    cases = ["pglib_opf_case300_ieee.m", "pglib_opf_case1354_pegase.m"]

    for case_file, model_name, problem_type in product(cases, models, problem_types):
        case_name = os.path.splitext(case_file)[0]
        case_path = os.path.join("data", case_file)

        print(f"\n{'=' * 60}")
        print(f"case={case_file}  model={model_name}  problem={problem_type}")
        print(f"{'=' * 60}")

        loaded = load_best_config(model_name, problem_type, case_name)
        if loaded is None:
            continue

        best_config: dict = loaded["config"]
        print(f"Best val_loss from search: {loaded['best_val_loss']:.6f}")
        print(f"Best config: {best_config}")

        results_filename = (
            f"results_{model_name}_{problem_type}_{MODE}_{case_name}.json"
        )
        results_file = os.path.join(TRAIN_RESULTS_DIR, results_filename)

        command = get_command(
            model_name, problem_type, case_path, best_config, results_file
        )
        print(f"Command: {command}")
        os.system(command)

    print("\nAll training runs complete.")
