"""
Hyperparameter search for E2ELR using Ray Tune + Optuna + ASHA.

Usage:
    python hparam_search.py --case data/pglib_opf_case300_ieee.m \
        --model e2elr --problem ed --mode ssl \
        --num_samples 50 --max_epochs_per_trial 100 \
        --skip_solve

Search space:
    n_layers    : [2, 3, 4]
    hidden_dim  : [128, 256, 512, 1024]
    lam         : [1.0]
    lr          : log-uniform [1e-4, 1e-2]
    batch_size  : [32, 64, 128, 256]

Metric optimised: best validation loss (always available, no LP required).
"""

import argparse
import json
import os
import time

import numpy as np
import ray
import torch
from data_utils import (
    compute_B_matrix,
    compute_reserve_params,
    extract_case_data,
    generate_instances,
    parse_matpower,
    solve_all_instances,
)
from models import DNNModel, E2ELRDCModel, E2ELRModel
from ray import tune
from ray.tune import report
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from train import build_datasets, train_model

# ---------------------------------------------------------------------------
# Trial function
# ---------------------------------------------------------------------------


def run_trial(
    config: dict,
    *,
    datasets_ref: dict,
    case_ref: dict,
    tensors_ref: dict[str, torch.Tensor],
    cli_args,
):
    """Single Ray Tune trial: build model → train → report val_loss."""

    datasets = datasets_ref
    case = case_ref
    tensors = tensors_ref

    device = cli_args.device

    # Move tensors to device
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    t = {k: _t(v) for k, v in tensors.items()}

    # Move datasets to device
    device_datasets = {}
    for name, ds in datasets.items():
        device_datasets[name] = torch.utils.data.TensorDataset(
            *[tensor.to(device) for tensor in ds.tensors]
        )

    # r_max numpy array (for model constructor)
    r_max_np = tensors["r_max_np"]

    # Build model
    n_layers = config["n_layers"]
    hidden_dim = config["hidden_dim"]

    if cli_args.model == "dnn":
        model = DNNModel(
            n_bus=case["n_bus"],
            n_gen=case["n_gen"],
            pg_max=case["pg_max"],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            B_pinv=tensors["B_pinv_np"],
            gen_bus_idx=case["gen_bus_idx"],
        )
    elif cli_args.model == "e2elr":
        model = E2ELRModel(
            n_bus=case["n_bus"],
            n_gen=case["n_gen"],
            pg_max=case["pg_max"],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            problem_type=cli_args.problem,
            r_max=r_max_np,
            B_pinv=tensors["B_pinv_np"],
            gen_bus_idx=case["gen_bus_idx"],
        )
    elif cli_args.model == "e2elrdc":
        model = E2ELRDCModel(
            n_bus=case["n_bus"],
            n_gen=case["n_gen"],
            pg_max=case["pg_max"],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            problem_type=cli_args.problem,
            r_max=r_max_np,
            B=tensors["B_bus_sparse"],
            gen_bus_idx=case["gen_bus_idx"],
        )
    else:
        raise ValueError(f"Unknown model: {cli_args.model}")

    model = model.to(device)

    lam = config["lam"]

    def epoch_callback(epoch, train_loss, val_loss, elapsed_min):
        report({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "training_time_min": elapsed_min,
        })

    _, history = train_model(
        model,
        device_datasets,
        case,
        cli_args.problem,
        mode=cli_args.mode,
        lam=lam,
        mu=lam,  # mu = lambda per paper convention
        b_branch_t=t["b_branch_t"],
        branch_from=tensors["branch_from"],
        branch_to=tensors["branch_to"],
        branch_rate_t=t["branch_rate_t"],
        B_bus_t=t["B_bus_t"],
        gen_bus_idx_t=t["gen_bus_idx_t"],
        cost_coef_t=t["cost_coef_t"],
        pg_max_t=t["pg_max_t"],
        r_max_t=t["r_max_t"],
        lr=config["lr"],
        weight_decay=1e-6,
        batch_size_train=config["batch_size"],
        batch_size_eval=256,
        max_epochs=cli_args.max_epochs_per_trial,
        patience=cli_args.patience,
        lr_patience=10,
        max_time_min=cli_args.max_time_per_trial_min,
        device=device,
        verbose=False,
        epoch_callback=epoch_callback,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for E2ELR via Ray Tune + Optuna + ASHA"
    )

    # ---- Data / problem (mirrored from main.py) ----
    parser.add_argument("--case", type=str, default="data/pglib_opf_case300_ieee.m")
    parser.add_argument("--problem", type=str, default="ed", choices=["ed", "edr"])
    parser.add_argument("--n_instances", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_solve",
        action="store_true",
        help="Skip LP solving (recommended with SSL mode)",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # ---- Model ----
    parser.add_argument(
        "--model", type=str, default="e2elr", choices=["dnn", "e2elr", "e2elrdc"]
    )
    parser.add_argument("--mode", type=str, default="ssl", choices=["sl", "ssl"])

    # ---- Training limits per trial ----
    parser.add_argument(
        "--max_epochs_per_trial",
        type=int,
        default=100,
        help="Maximum epochs allowed per trial",
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early-stopping patience per trial"
    )
    parser.add_argument(
        "--max_time_per_trial_min",
        type=float,
        default=30.0,
        help="Max wall-clock minutes per trial",
    )

    # ---- Ray Tune ----
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Total number of hyperparameter trials",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=4,
        help="Max concurrent trials (Optuna concurrency limiter)",
    )
    parser.add_argument(
        "--num_cpus", type=int, default=1, help="CPUs allocated per trial"
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=0.0,
        help="GPUs allocated per trial (fractional allowed)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ray_results",
        help="Directory where Ray Tune stores trial results",
    )

    # ---- Device ----
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: cpu, cuda, mps, or auto"
    )

    args = parser.parse_args()

    # Device resolution
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # 1. Load case data
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Loading case: {args.case}")
    print(f"{'=' * 60}")

    raw = parse_matpower(args.case)
    case = extract_case_data(raw)
    print(
        f"  Buses: {case['n_bus']}, Generators: {case['n_gen']}, "
        f"Branches: {case['n_branch']}"
    )

    # -----------------------------------------------------------------------
    # 2. B-matrix
    # -----------------------------------------------------------------------
    print("Computing B-matrix...")
    B_bus, B_pinv, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack = (
        compute_B_matrix(case)
    )

    # -----------------------------------------------------------------------
    # 3. Generate instances
    # -----------------------------------------------------------------------
    print(f"Generating {args.n_instances} {args.problem.upper()} instances...")
    instances = generate_instances(case, args.n_instances, args.problem, seed=args.seed)

    r_max_np = None
    if args.problem == "edr":
        alpha_r, r_max_np = compute_reserve_params(case)
        instances["r_max"] = r_max_np
        print(f"  alpha_r = {alpha_r:.4f}")

    # -----------------------------------------------------------------------
    # 4. Optional LP solve (needed for SL mode)
    # -----------------------------------------------------------------------
    pg_star, obj_star = None, None

    case_name = os.path.splitext(os.path.basename(args.case))[0]
    gt_path = os.path.join(
        args.checkpoint_dir,
        f"gt_{case_name}_{args.problem}_n{args.n_instances}_s{args.seed}.npz",
    )

    if args.mode == "sl" or not args.skip_solve:
        if os.path.isfile(gt_path):
            print(f"Loading ground truth from: {gt_path}")
            gt = np.load(gt_path)
            pg_star, obj_star = gt["pg_star"], gt["obj_star"]
        else:
            print("Solving instances with CVXPY (this may take a while)...")
            t0 = time.time()
            pg_star, obj_star = solve_all_instances(
                instances,
                case,
                b_branch,
                B_reduced_csc,
                F_theta,
                C_g_r,
                non_slack,
                problem_type=args.problem,
                M_th=15.0,
                verbose=True,
            )
            print(f"  Done in {(time.time() - t0) / 60:.1f} min")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            np.savez(gt_path, pg_star=pg_star, obj_star=obj_star)
            print(f"  Saved to: {gt_path}")
    else:
        print("Skipping LP solving (SSL mode + --skip_solve).")

    # -----------------------------------------------------------------------
    # 5. Build datasets (once, shared across all trials)
    # -----------------------------------------------------------------------
    print("Building datasets...")
    # Build on CPU so tensors can be moved to any device inside each trial
    datasets = build_datasets(
        instances, pg_star, obj_star, case, args.problem, device="cpu"
    )
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} instances")

    # -----------------------------------------------------------------------
    # 6. Pre-build fixed tensors (CPU) shared across all trials
    # -----------------------------------------------------------------------
    r_max_arr = instances.get("r_max", np.zeros(case["n_gen"], dtype=np.float32))

    tensors = {
        # numpy/scipy objects needed for model constructors
        "B_pinv_np": B_pinv,
        "B_bus_sparse": B_bus,
        "r_max_np": r_max_np,
        # index arrays (stay as numpy, used as integer indices)
        "branch_from": case["branch_from"],
        "branch_to": case["branch_to"],
        # torch tensors (moved to device inside each trial)
        "b_branch_t": torch.tensor(b_branch, dtype=torch.float32),
        "branch_rate_t": torch.tensor(case["branch_rate"], dtype=torch.float32),
        "B_bus_t": torch.tensor(B_bus.toarray(), dtype=torch.float32),
        "gen_bus_idx_t": torch.tensor(case["gen_bus_idx"], dtype=torch.long),
        "cost_coef_t": torch.tensor(case["cost_coef"], dtype=torch.float32),
        "pg_max_t": torch.tensor(case["pg_max"], dtype=torch.float32),
        "r_max_t": torch.tensor(r_max_arr, dtype=torch.float32),
    }

    # -----------------------------------------------------------------------
    # 7. Initialise Ray and upload shared data to object store
    # -----------------------------------------------------------------------
    print("\nInitialising Ray...")
    ray.init(ignore_reinit_error=True)

    datasets_ref = ray.put(datasets)
    case_ref = ray.put(case)
    tensors_ref = ray.put(tensors)

    print("  Shared data uploaded to Ray object store.")

    # -----------------------------------------------------------------------
    # 8. Search space
    # -----------------------------------------------------------------------
    search_space = {
        "n_layers": tune.choice([2, 3, 4]),
        "hidden_dim": tune.choice([128, 256, 512, 1024]),
        "lam": tune.choice([1.0]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256]),
    }

    # -----------------------------------------------------------------------
    # 9. Scheduler + search algorithm
    # -----------------------------------------------------------------------
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="val_loss",
        mode="min",
        max_t=args.max_epochs_per_trial,
        grace_period=min(10, args.max_epochs_per_trial),
        reduction_factor=2,
    )

    search_alg = OptunaSearch(metric="val_loss", mode="min")
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)

    # -----------------------------------------------------------------------
    # 10. Run search
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Starting hyperparameter search")
    print(f"  Model:    {args.model.upper()}")
    print(f"  Problem:  {args.problem.upper()}")
    print(f"  Mode:     {args.mode.upper()}")
    print(f"  Trials:   {args.num_samples}")
    print(f"  Max epochs/trial: {args.max_epochs_per_trial}")
    print(f"  Results dir: {args.results_dir}")
    print(f"{'=' * 60}\n")

    analysis = tune.run(
        tune.with_parameters(
            run_trial,
            datasets_ref=datasets_ref,
            case_ref=case_ref,
            tensors_ref=tensors_ref,
            cli_args=args,
        ),
        config=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": args.num_cpus, "gpu": args.num_gpus},
        storage_path=os.path.abspath(args.results_dir),
        name=f"hparam_{args.model}_{args.problem}_{args.mode}",
        verbose=1,
        raise_on_failed_trial=False,
    )

    # -----------------------------------------------------------------------
    # 11. Report results
    # -----------------------------------------------------------------------
    best_trial = analysis.get_best_trial("val_loss", mode="min", scope="all")

    if best_trial is None:
        print("No successful trials found.")
        ray.shutdown()
        raise RuntimeError("Hyperparameter search failed: no valid trials.")

    best_config = best_trial.config
    best_val_loss = best_trial.last_result["val_loss"]

    print(f"\n{'=' * 60}")
    print("Hyperparameter search complete!")
    print(f"  Best val_loss : {best_val_loss:.6f}")
    print("  Best config   :")
    for k, v in best_config.items():
        print(f"    {k:>12s} = {v}")
    print(f"{'=' * 60}")

    # Save best config to JSON
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir,
        f"best_config_{args.model}_{args.problem}_{args.mode}.json",
    )
    with open(out_path, "w") as f:
        json.dump({"best_val_loss": best_val_loss, "config": best_config}, f, indent=2)
    print(f"\nBest config saved to: {out_path}")

    # Print top-10 trials sorted by val_loss
    df = analysis.results_df
    if "val_loss" in df.columns:
        top = df.sort_values("val_loss").head(10)
        cols = ["val_loss"] + [
            c
            for c in [
                "config/n_layers",
                "config/hidden_dim",
                "config/lam",
                "config/lr",
                "config/batch_size",
                "training_time_min",
            ]
            if c in df.columns
        ]
        print("\nTop-10 trials:")
        print(top[cols].to_string(index=False))

    ray.shutdown()
    print("\nDone.")
