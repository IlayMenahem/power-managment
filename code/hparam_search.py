"""
Hyperparameter search for E2ELR using Ray Tune + Optuna + ASHA.

Usage:
    python hparam_search.py --case data/pglib_opf_case300_ieee.m \
        --model e2elr e2elrdc dnn --problem ed --mode ssl \
        --num_samples 75 --max_epochs_per_trial 100 \
        --skip_solve

    # Connect to an existing Ray cluster (e.g. started by run_all.py):
    python hparam_search.py ... --ray_address auto

Search space:
    model       : values of --model flag
    n_layers    : [2, 3, 4]
    hidden_dim  : [128, 256, 512, 1024]
    lr          : log-uniform [1e-4, 1e-2]
    batch_size  : [64, 128, 256]

Fixed hyper-parameters (not searched):
    lam         : set via --lam (default: 1.0 for ssl, 1e-4 for sl)

Metric optimised: validation loss (always available, no LP required).
The best config per model is saved to
    <results_dir>/best_config_{model}_{problem}_{mode}_{case}.json
"""

import argparse
import json
import os

import numpy as np
import ray
import torch
from data_utils import (
    Case,
    generate_instances,
    ground_truth_path,
    load_case,
    load_or_solve_ground_truth,
)
from models import ARCHITECTURES, DispatchProxy
from ray import tune
from ray.tune import report
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import TensorDataset
from train import DEFAULT_LAM, GridTensors, build_datasets, resolve_device, train_model


def run_trial(
    config: dict,
    *,
    datasets: dict[str, TensorDataset],
    case: Case,
    r_max: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Single Ray Tune trial: build model -> train -> report val_loss every epoch."""
    device = args.device
    device_datasets = {
        name: TensorDataset(*(t.to(device) for t in ds.tensors))
        for name, ds in datasets.items()
    }
    model = DispatchProxy(
        config["model"], case, r_max, config["hidden_dim"], config["n_layers"]
    ).to(device)

    def epoch_callback(
        epoch: int, train_loss: float, val_loss: float, elapsed_min: float
    ) -> None:
        report(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "training_time_min": elapsed_min,
            }
        )

    train_model(
        model,
        device_datasets,
        GridTensors.from_case(case, r_max, device),
        mode=args.mode,
        lam=args.lam,
        lr=config["lr"],
        batch_size_train=config["batch_size"],
        max_epochs=args.max_epochs_per_trial,
        patience=args.patience,
        max_time_min=args.max_time_per_trial_min,
        verbose=False,
        epoch_callback=epoch_callback,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for E2ELR via Ray Tune + Optuna + ASHA"
    )
    # Data / problem
    parser.add_argument("--case", type=str, default="data/pglib_opf_case300_ieee.m")
    parser.add_argument("--problem", type=str, default="ed", choices=["ed", "edr"])
    parser.add_argument("--n_instances", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_solve",
        action="store_true",
        help="Do not solve the LPs if no ground-truth checkpoint exists (SSL only)",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    # Models (one or more) and training mode
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=list(ARCHITECTURES),
        choices=ARCHITECTURES,
        help="One or more model architectures to search over",
    )
    parser.add_argument("--mode", type=str, default="ssl", choices=["sl", "ssl"])
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Constraint penalty weight (default: 1.0 for ssl, 1e-4 for sl)",
    )
    # Training limits per trial
    parser.add_argument("--max_epochs_per_trial", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_time_per_trial_min", type=float, default=30.0)
    # Ray Tune
    parser.add_argument("--num_samples", type=int, default=75)
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=float, default=0.0)
    parser.add_argument("--results_dir", type=str, default="ray_results")
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help=(
            "Address of an existing Ray cluster to connect to "
            "(e.g. 'auto' or '127.0.0.1:6379'). "
            "If not set, a new local cluster is started and shut down on exit."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu, cuda, or auto (MPS lacks float64)",
    )

    args = parser.parse_args()
    if args.lam is None:
        args.lam = DEFAULT_LAM[args.mode]
    args.device = resolve_device(args.device)
    return args


def save_best_configs(
    analysis: tune.ExperimentAnalysis, args: argparse.Namespace, case: Case
) -> None:
    """Save the trial with the lowest final val_loss for each model."""
    config_keys = ["model", "n_layers", "hidden_dim", "lr", "batch_size"]
    df = analysis.results_df.dropna(subset=["val_loss"])
    os.makedirs(args.results_dir, exist_ok=True)

    for model_name in args.model:
        model_df = df[df["config/model"] == model_name]
        if model_df.empty:
            print(f"  [{model_name}] No successful trials found.")
            continue
        best = model_df.loc[model_df["val_loss"].idxmin()]
        best_config = {k: best[f"config/{k}"] for k in config_keys}
        print(f"\n  [{model_name.upper()}] Best val_loss: {best['val_loss']:.6f}")
        for k, v in best_config.items():
            print(f"    {k:>12s} = {v}")

        out_path = os.path.join(
            args.results_dir,
            f"best_config_{model_name}_{args.problem}_{args.mode}_{case.name}.json",
        )
        with open(out_path, "w") as f:
            json.dump(
                {"best_val_loss": float(best["val_loss"]), "config": best_config},
                f,
                indent=2,
                default=lambda v: v.item(),  # numpy scalars from the DataFrame
            )
        print(f"    Saved to: {out_path}")

    top_cols = ["val_loss", *(f"config/{k}" for k in config_keys), "training_time_min"]
    print("\nTop-10 trials overall:")
    print(df.sort_values("val_loss").head(10)[top_cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    print(f"Using device: {args.device}")
    print(f"Models in search space: {args.model}")
    print(f"Fixed lam: {args.lam}")
    torch.manual_seed(args.seed)

    # 1. Data shared across all trials
    case = load_case(args.case)
    print(f"Case {case.summary()}")
    instances = generate_instances(case, args.n_instances, args.problem, args.seed)
    path = ground_truth_path(
        args.checkpoint_dir, case, args.problem, args.n_instances, args.seed
    )
    ground_truth = load_or_solve_ground_truth(
        case, instances, path, solve_if_missing=args.mode == "sl" or not args.skip_solve
    )
    datasets = build_datasets(instances, ground_truth, device="cpu")

    # 2. Connect to or start a Ray cluster
    owns_cluster = args.ray_address is None
    print(
        "\nStarting new Ray cluster"
        if owns_cluster
        else f"\nConnecting to Ray at {args.ray_address}"
    )
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    # 3. Search space, scheduler and search algorithm
    search_space = {
        "model": tune.choice(args.model),
        "n_layers": tune.choice([2, 3, 4]),
        "hidden_dim": tune.choice([128, 256, 512, 1024]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([64, 128, 256]),
    }
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="val_loss",
        mode="min",
        max_t=args.max_epochs_per_trial,
        grace_period=min(10, args.max_epochs_per_trial),
        reduction_factor=2,
    )
    search_alg = ConcurrencyLimiter(
        OptunaSearch(metric="val_loss", mode="min"), max_concurrent=args.max_concurrent
    )

    # 4. Run search; with_parameters stores the shared data in the Ray object store
    print(
        f"\n{'=' * 60}\nStarting hyperparameter search: models={args.model} "
        f"problem={args.problem.upper()} mode={args.mode.upper()} "
        f"trials={args.num_samples}\n{'=' * 60}"
    )
    analysis = tune.run(
        tune.with_parameters(
            run_trial, datasets=datasets, case=case, r_max=instances.r_max, args=args
        ),
        config=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": args.num_cpus, "gpu": args.num_gpus},
        storage_path=os.path.abspath(args.results_dir),
        name=f"hparam_{'_'.join(args.model)}_{args.problem}_{args.mode}",
        verbose=1,
        raise_on_failed_trial=False,
    )

    # 5. Report results
    print(f"\n{'=' * 60}\nHyperparameter search complete!")
    save_best_configs(analysis, args, case)
    print(f"{'=' * 60}")

    if owns_cluster:
        ray.shutdown()
    print("\nDone.")


if __name__ == "__main__":
    main()
