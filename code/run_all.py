"""
Run the hyperparameter tuning for all model architectures, problem types, and cases.

Strategy
--------
- One persistent Ray cluster is started here and shared across all child invocations
  of hparam_search.py, avoiding per-run init/shutdown overhead.
- Each (case, problem_type, model) triple gets its own hparam_search.py run,
  so results are saved separately per model as before.
- The outer loop is product(cases, problem_types, models) — 12 sequential runs,
  each connecting to the shared cluster instead of starting a fresh one.
"""

import os
import sys
from itertools import product

import ray
import torch

NUM_TRIALS = 1
NUM_PROCESSES = 6


def get_command(model: str, problem_type: str, case: str, ray_address: str) -> str:
    num_gpus = 1.0 / NUM_PROCESSES if torch.cuda.is_available() else 0
    num_cpus = max(1, (os.cpu_count() or 1) // NUM_PROCESSES)
    case_path = os.path.join("data", case)

    return (
        f"{sys.executable} hparam_search.py"
        f" --skip_solve"
        f" --model {model}"
        f" --problem {problem_type}"
        f" --case {case_path}"
        f" --num_samples {NUM_TRIALS}"
        f" --num_cpus {num_cpus}"
        f" --num_gpus {num_gpus}"
        f" --max_concurrent {NUM_PROCESSES}"
        f" --ray_address {ray_address}"
    )


if __name__ == "__main__":
    models = ["dnn", "e2elr", "e2elrdc"]
    problem_types = ["ed", "edr"]
    cases = ["pglib_opf_case300_ieee.m", "pglib_opf_case1354_pegase.m"]

    ctx = ray.init(ignore_reinit_error=True)
    ray_address = ctx.address_info["address"]
    print(f"Ray cluster started at: {ray_address}\n")

    try:
        for case, model, problem_type in product(cases, models, problem_types):
            command = get_command(model, problem_type, case, ray_address)
            print(f"{'=' * 60}")
            print(f"Running: case={case}  model={model}  problem={problem_type}")
            print(f"Command: {command}")
            print(f"{'=' * 60}")
            os.system(command)
    finally:
        ray.shutdown()
        print("\nRay cluster shut down. Done.")
