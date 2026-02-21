"""
Run the hyperparameter tuning or training of all the models, problem types, and cases.
"""

def get_command(model: str, problem_type: str, case: str) -> str:
    return f"python hparam_search.py --model {model} --problem {problem_type} --case {case} --num_samples 25 --num_cpus 1"


if __name__ == "__main__":
    from itertools import product
    import os

    models = ["dnn", "e2elr", "e2elrdc"]
    problem_types = ["ed", "edr"]
    cases = ["pglib_opf_case300_ieee.m", "pglib_opf_case1354_pegase.m"]

    for model, problem_type, case in product(models, problem_types, cases):
        command = get_command(model, problem_type, case)
        print(f"Running command: {command}")
        os.system(command)