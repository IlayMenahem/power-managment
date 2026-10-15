"""
Run the hyperparameter search for every (case, model, problem) combination.

Strategy
--------
- One persistent Ray cluster is started here and shared across all child
  invocations of hparam_search.py, avoiding per-run init/shutdown overhead.
- Each (case, model, problem) triple gets its own hparam_search.py run, so the
  best config is saved separately per model.
"""

import os
import subprocess
import sys
from itertools import product

import ray
import torch

NUM_TRIALS = 25
NUM_PROCESSES = 6
MODELS = ["dnn", "e2elr", "e2elrdc"]
PROBLEMS = ["edr"]
CASES = ["pglib_opf_case300_ieee.m", "pglib_opf_case1354_pegase.m"]


def get_command(model: str, problem: str, case: str, ray_address: str) -> list[str]:
    num_gpus = 1.0 / NUM_PROCESSES if torch.cuda.is_available() else 0
    num_cpus = max(1, (os.cpu_count() or 1) // NUM_PROCESSES)
    return [
        sys.executable,
        "hparam_search.py",
        "--skip_solve",
        "--model", model,
        "--problem", problem,
        "--case", os.path.join("data", case),
        "--num_samples", str(NUM_TRIALS),
        "--num_cpus", str(num_cpus),
        "--num_gpus", str(num_gpus),
        "--max_concurrent", str(NUM_PROCESSES),
        "--ray_address", ray_address,
    ]  # fmt: skip


if __name__ == "__main__":
    ctx = ray.init(ignore_reinit_error=True)
    ray_address = ctx.address_info["address"]
    print(f"Ray cluster started at: {ray_address}\n")

    failed = []
    try:
        for case, model, problem in product(CASES, MODELS, PROBLEMS):
            command = get_command(model, problem, case, ray_address)
            print(f"{'=' * 60}")
            print(f"Running: case={case}  model={model}  problem={problem}")
            print(f"Command: {' '.join(command)}")
            print(f"{'=' * 60}")
            if subprocess.run(command).returncode != 0:
                failed.append((case, model, problem))
    finally:
        ray.shutdown()
        print("\nRay cluster shut down.")

    print(f"Failed runs: {failed}" if failed else "All runs succeeded.")
