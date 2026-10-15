"""
Main entry point for E2ELR: End-to-End Feasible Optimization Proxies
for Large-Scale Economic Dispatch.

Usage:
    python main.py --case data/pglib_opf_case300_ieee.m --model e2elr --problem ed --mode ssl
    python main.py --case data/pglib_opf_case300_ieee.m --model dnn --problem edr --mode sl
    python main.py --case data/pglib_opf_case300_ieee.m --model e2elr --problem edr --line_limits hard

Reference: arXiv 2304.11726v2
"""

import argparse
import json
import os

import mlx.core as mx
import numpy as np
from data_utils import LINE_LIMITS, compute_reserve_params, load_case, load_dataset
from mlx.utils import tree_flatten
from models import ARCHITECTURES, DispatchProxy
from benchmarks import inference_time_ms, repair_vs_projection
from train import (
    GridTensors,
    build_datasets,
    default_lam,
    evaluate_model,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2ELR: Economic Dispatch Optimization Proxy"
    )
    # Data and problem
    parser.add_argument(
        "--case",
        type=str,
        default="data/pglib_opf_case300_ieee.m",
        help="Path to MATPOWER .m case file",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="ed",
        choices=["ed", "edr"],
        help="Problem type: ed (no reserves) or edr (with reserves)",
    )
    parser.add_argument(
        "--line_limits",
        type=str,
        default="soft",
        choices=LINE_LIMITS,
        help="soft: thermal limits penalized (paper); hard: dataset of instances "
        "feasible with hard thermal limits (rejection sampling)",
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=50000,
        help="Total number of instances to generate",
    )
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="e2elr",
        choices=ARCHITECTURES,
        help="Model architecture",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="Number of hidden layers in DNN backbone",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension of DNN backbone"
    )
    # Training
    parser.add_argument(
        "--mode",
        type=str,
        default="ssl",
        choices=["sl", "ssl"],
        help="Training mode: sl (supervised) or ssl (self-supervised)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Constraint penalty weight (default: 1.0 for ssl, 1e-4 for sl, 0 for "
        "e2elr with ssl)",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500, help="Maximum training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--max_time", type=float, default=150.0, help="Max training time in minutes"
    )
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_solve",
        action="store_true",
        help="Do not solve the LPs if no ground-truth checkpoint exists (SSL, soft "
        "limits only)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving/loading ground truth solutions",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Optional path to save evaluation results as JSON (e.g. results/run.json)",
    )
    return parser.parse_args()


def print_results(
    results: dict[str, float], history: dict, has_ground_truth: bool, title: str
) -> None:
    print(f"\n--- Results ({title}) ---")
    print(f"  Test Loss:               {results['test_loss']:.6f}")
    if has_ground_truth:
        print(f"  Mean Optimality Gap:     {results['mean_gap_pct']:.4f}%")
        print(f"  SGM Optimality Gap:      {results['sgm_gap_pct']:.4f}%")
    else:
        print("  Optimality Gap:          N/A (no ground truth)")
    print(f"  Feasibility Rate:        {results['feasibility_rate_pct']:.2f}%")
    print(
        f"  Power Balance:           {results['pb_feasible_pct']:.2f}% feasible, "
        f"violation {results['pb_violation_sgm_pu']:.4g} p.u. (SGM of infeasible)"
    )
    print(
        f"  Reserves:                {results['reserve_feasible_pct']:.2f}% feasible, "
        f"shortage {results['reserve_violation_sgm_pu']:.4g} p.u. (SGM of infeasible)"
    )
    print(f"  Line Feasibility Rate:   {results['line_feasibility_rate_pct']:.2f}%")
    print(f"  Mean Thermal Violation:  {results['mean_thermal_violation_pu']:.6f} p.u.")
    print(f"  Mean Max DC Residual:    {results['mean_max_dc_residual_pu']:.2e} p.u.")
    print(f"  Inference Time:          {results['inference_ms_per_256']:.2f} ms / 256")
    if "repair_us" in results:
        print(
            f"  Repair vs Projection:    {results['repair_us']:.1f} us vs "
            f"{results['projection_us']:.1f} us per instance "
            f"({results['projection_speedup']:.0f}x)"
        )
    print(f"  Best Val Loss:           {history['best_val_loss']:.6f}")
    print(f"  Training Time:           {history['training_time_min']:.1f} min")


def save_results(
    path: str, args: argparse.Namespace, results: dict[str, float], history: dict
) -> None:
    run_record = {
        "case": args.case,
        "model": args.model,
        "problem": args.problem,
        "line_limits": args.line_limits,
        "mode": args.mode,
        "config": {
            "n_layers": args.n_layers,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "lam": args.lam,
        },
        "training_time_min": history["training_time_min"],
        "best_val_loss": history["best_val_loss"],
        "results": {k: float(v) for k, v in results.items()},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"\nResults saved to: {path}")


def main() -> None:
    args = parse_args()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Case
    print(f"\n{'=' * 60}\nLoading case: {args.case}\n{'=' * 60}")
    case = load_case(args.case)
    print(f"  {case.summary()}")
    print(
        f"  Total ref demand: {case.pd_ref.sum():.2f} p.u. "
        f"({case.pd_ref.sum() * case.base_mva:.0f} MW)"
    )
    print(
        f"  Total gen capacity: {case.pg_max.sum():.2f} p.u. "
        f"({case.pg_max.sum() * case.base_mva:.0f} MW)"
    )

    # 2. Instances and ground truth (needed for SL and for optimality gaps)
    print(
        f"\n{args.n_instances} {args.problem.upper()} instances, "
        f"{args.line_limits} thermal limits"
    )
    instances, ground_truth = load_dataset(
        case,
        args.problem,
        args.n_instances,
        args.seed,
        args.line_limits,
        args.checkpoint_dir,
        solve_if_missing=args.mode == "sl" or not args.skip_solve,
    )
    if args.problem == "edr":
        print(f"  alpha_r = {compute_reserve_params(case)[0]:.4f}")
        print(f"  R range: [{instances.R.min():.4f}, {instances.R.max():.4f}] p.u.")
    if ground_truth is not None:
        obj_star = ground_truth[1]
        valid = obj_star[~np.isnan(obj_star)]
        print(f"  Valid solutions: {len(valid)}/{len(obj_star)}")
        if len(valid):
            print(f"  Objective range: [{valid.min():.2f}, {valid.max():.2f}]")

    # 3. Datasets and model
    datasets = build_datasets(instances, ground_truth)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} instances")

    model = DispatchProxy(
        args.model, case, instances.r_max, args.hidden_dim, args.n_layers
    )
    n_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"\nModel: {args.model.upper()} ({n_params:,} parameters)")

    grid = GridTensors.from_case(case, instances.r_max)
    if args.lam is None:
        args.lam = default_lam(args.mode, args.model)
    lam = args.lam

    # 4. Train
    print(
        f"\n{'=' * 60}\nTraining {args.model.upper()} with {args.mode.upper()}, "
        f"lambda={lam}\n{'=' * 60}"
    )
    history = train_model(
        model,
        datasets,
        grid,
        mode=args.mode,
        lam=lam,
        lr=args.lr,
        batch_size_train=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_time_min=args.max_time,
    )

    # 5. Evaluate
    print(f"\n{'=' * 60}\nEvaluating on test set\n{'=' * 60}")
    results = evaluate_model(model, datasets["test"], grid, mode=args.mode, lam=lam)
    results["inference_ms_per_256"] = inference_time_ms(model, datasets["test"])
    if args.model != "dnn":
        results.update(
            repair_vs_projection(model, datasets["test"], case, instances.r_max)
        )
    title = f"{args.model.upper()}, {args.mode.upper()}, {args.problem.upper()}"
    print_results(results, history, ground_truth is not None, title)

    if args.results_file is not None:
        save_results(args.results_file, args, results, history)
    print("\nDone.")


if __name__ == "__main__":
    main()
