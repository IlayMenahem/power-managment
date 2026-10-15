"""
Train every (case, model, problem) combination with the paper's training
settings (Appendix B): Adam with lr 1e-2 and weight decay 1e-6, batch size 64,
no gradient clipping, and one architecture from the paper's grid, 3 layers of
256 units (no hyperparameter search). These are the defaults of main.py.

Test metrics are written to
    train_results/results_{model}_{problem}_{line_limits}_{mode}_{case_name}.json
"""

import os
import subprocess
import sys
from itertools import product

MODELS = ["dnn", "e2elr"]
PROBLEMS = ["edr"]
LINE_LIMITS = "soft"  # "hard": instances feasible with hard thermal limits
CASES = ["pglib_opf_case300_ieee.m", "pglib_opf_case1354_pegase.m"]
MODE = "ssl"
TRAIN_RESULTS_DIR = "train_results"


def get_command(model: str, problem: str, case_path: str, results_file: str) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--model", model,
        "--problem", problem,
        "--case", case_path,
        "--mode", MODE,
        "--line_limits", LINE_LIMITS,
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

        results_file = os.path.join(
            TRAIN_RESULTS_DIR,
            f"results_{model}_{problem}_{LINE_LIMITS}_{MODE}_{case_name}.json",
        )
        command = get_command(
            model, problem, os.path.join("data", case_file), results_file
        )
        print(f"Command: {' '.join(command)}")
        if subprocess.run(command).returncode != 0:
            failed.append((case_file, model, problem))

    print(f"\nFailed runs: {failed}" if failed else "\nAll training runs complete.")
