"""
Run the hyperparameter tuning or training of all the models, problem types, and cases.
"""

import os
from itertools import product


def get_command(model: str, problem_type: str, case: str) -> str:
    num_processes = 22

    num_gpus = 1.0 / num_processes
    num_cpus = max(1, os.cpu_count() // num_processes)

    case = os.path.join("data", case)

    return f"python hparam_search.py --skip_solve --model {model} --problem {problem_type} --case {case} --num_samples 25 --num_cpus {num_cpus} --num_gpus {num_gpus} --max_concurrent {num_processes}"


if __name__ == "__main__":
    models = ["dnn", "e2elr", "e2elrdc"]
    problem_types = ["ed", "edr"]
    cases = ["pglib_opf_case300_ieee.m", "pglib_opf_case1354_pegase.m"]

    for model, problem_type, case in product(cases, models, problem_types):
        command = get_command(model, problem_type, case)
        print(f"Running command: {command}")
        print("and waiting for it to finish...")
        os.system(command)
