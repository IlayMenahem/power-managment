"""
Main entry point for E2ELR: End-to-End Feasible Optimization Proxies
for Large-Scale Economic Dispatch.

Usage:
    python main.py --case data/pglib_opf_case300_ieee.m --model e2elr --problem ed --mode ssl
    python main.py --case data/pglib_opf_case300_ieee.m --model dnn --problem edr --mode sl

Reference: arXiv 2304.11726v2
"""

import argparse
import os
import time
import numpy as np
import torch

from data_utils import (
    parse_matpower, extract_case_data, compute_B_matrix,
    generate_instances, solve_all_instances,
)
from models import DNNModel, E2ELRModel, E2ELRDCModel
from train import (
    build_datasets, train_model, evaluate_model,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2ELR: Economic Dispatch Optimization Proxy")

    # Data and problem
    parser.add_argument("--case", type=str, default="data/pglib_opf_case300_ieee.m",
                        help="Path to MATPOWER .m case file")
    parser.add_argument("--problem", type=str, default="ed", choices=["ed", "edr"],
                        help="Problem type: ed (no reserves) or edr (with reserves)")
    parser.add_argument("--n_instances", type=int, default=50000,
                        help="Total number of instances to generate")

    # Model
    parser.add_argument("--model", type=str, default="e2elr", choices=["dnn", "e2elr", "e2elrdc"],
                        help="Model architecture")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Number of hidden layers in DNN backbone")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension of DNN backbone")

    # Training
    parser.add_argument("--mode", type=str, default="ssl", choices=["sl", "ssl"],
                        help="Training mode: sl (supervised) or ssl (self-supervised)")
    parser.add_argument("--lam", type=float, default=None,
                        help="Constraint penalty weight (auto-set if None)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--max_time", type=float, default=150.0,
                        help="Max training time in minutes")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--skip_solve", action="store_true",
                        help="Skip LP solving (use with SSL mode)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for saving/loading ground truth solutions")

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # 1. Load case data
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading case: {args.case}")
    print(f"{'='*60}")

    raw = parse_matpower(args.case)
    case = extract_case_data(raw)

    print(f"  Buses: {case['n_bus']}, Generators: {case['n_gen']}, Branches: {case['n_branch']}")
    print(f"  Total ref demand: {case['pd_ref'].sum():.2f} p.u. "
          f"({case['pd_ref'].sum() * case['baseMVA']:.0f} MW)")
    print(f"  Total gen capacity: {case['pg_max'].sum():.2f} p.u. "
          f"({case['pg_max'].sum() * case['baseMVA']:.0f} MW)")

    # -----------------------------------------------------------------------
    # 2. Compute B-matrix
    # -----------------------------------------------------------------------
    print("\nComputing B-matrix...")
    t0 = time.time()
    B_bus, B_pinv, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack = compute_B_matrix(case)
    print(f"  B_pinv shape: {B_pinv.shape} — done in {time.time()-t0:.2f}s")

    # -----------------------------------------------------------------------
    # 3. Generate instances
    # -----------------------------------------------------------------------
    print(f"\nGenerating {args.n_instances} {args.problem.upper()} instances...")
    instances = generate_instances(case, args.n_instances, args.problem, seed=args.seed)

    if args.problem == "edr":
        from data_utils import compute_reserve_params
        alpha_r, r_max = compute_reserve_params(case)
        instances["r_max"] = r_max
        print(f"  alpha_r = {alpha_r:.4f}")
        print(f"  R range: [{instances['R_req'].min():.4f}, {instances['R_req'].max():.4f}] p.u.")

    # -----------------------------------------------------------------------
    # 4. Solve instances (for SL, or for evaluation gap computation)
    # -----------------------------------------------------------------------
    pg_star, obj_star = None, None

    # Derive a checkpoint filename from the case file and problem settings
    case_name = os.path.splitext(os.path.basename(args.case))[0]
    gt_checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"gt_{case_name}_{args.problem}_n{args.n_instances}_s{args.seed}.npz",
    )

    if args.mode == "sl" or not args.skip_solve:
        # Try loading ground truth from checkpoint
        if os.path.isfile(gt_checkpoint_path):
            print(f"\nLoading ground truth from checkpoint: {gt_checkpoint_path}")
            gt_data = np.load(gt_checkpoint_path)
            pg_star = gt_data["pg_star"]
            obj_star = gt_data["obj_star"]
            print(f"  Loaded pg_star {pg_star.shape}, obj_star {obj_star.shape}")
        else:
            print("\nSolving instances with CVXPY...")
            t0 = time.time()
            pg_star, obj_star = solve_all_instances(
                instances, case, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack,
                problem_type=args.problem, M_th=15.0, verbose=True,
            )
            print(f"  Solving done in {(time.time()-t0)/60:.1f} min")

            # Save ground truth checkpoint
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            np.savez(gt_checkpoint_path, pg_star=pg_star, obj_star=obj_star)
            print(f"  Ground truth saved to: {gt_checkpoint_path}")

        # Report stats
        valid = ~np.isnan(obj_star)
        print(f"  Valid solutions: {valid.sum()}/{len(obj_star)}")
        if valid.any():
            print(f"  Objective range: [{obj_star[valid].min():.2f}, {obj_star[valid].max():.2f}]")
    else:
        print("\nSkipping LP solving (SSL mode with --skip_solve).")

    # -----------------------------------------------------------------------
    # 5. Build datasets
    # -----------------------------------------------------------------------
    print("\nBuilding datasets...")
    datasets = build_datasets(instances, pg_star, obj_star, case, args.problem, device=device)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} instances")

    # -----------------------------------------------------------------------
    # 6. Build model
    # -----------------------------------------------------------------------
    r_max_np = instances["r_max"] if args.problem == "edr" else None

    if args.model == "dnn":
        model = DNNModel(
            n_bus=case["n_bus"], n_gen=case["n_gen"], pg_max=case["pg_max"],
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
            B_pinv=B_pinv, gen_bus_idx=case["gen_bus_idx"],
        )
    elif args.model == "e2elr":
        model = E2ELRModel(
            n_bus=case["n_bus"], n_gen=case["n_gen"], pg_max=case["pg_max"],
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
            problem_type=args.problem, r_max=r_max_np,
            B_pinv=B_pinv, gen_bus_idx=case["gen_bus_idx"],
        )
    elif args.model == "e2elrdc":
        model = E2ELRDCModel(
            n_bus=case["n_bus"], n_gen=case["n_gen"], pg_max=case["pg_max"],
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
            problem_type=args.problem, r_max=r_max_np,
            B=B_bus, gen_bus_idx=case["gen_bus_idx"],
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model.upper()} ({n_params:,} parameters)")

    # -----------------------------------------------------------------------
    # 7. Prepare tensors for loss computation
    # -----------------------------------------------------------------------
    b_branch_t = torch.tensor(b_branch, dtype=torch.float32, device=device)
    branch_from = case["branch_from"]   # int numpy array, used as index
    branch_to = case["branch_to"]       # int numpy array, used as index
    branch_rate_t = torch.tensor(case["branch_rate"], dtype=torch.float32, device=device)
    B_bus_t = torch.tensor(B_bus.toarray(), dtype=torch.float32, device=device)
    gen_bus_idx_t = torch.tensor(case["gen_bus_idx"], dtype=torch.long, device=device)
    cost_coef_t = torch.tensor(case["cost_coef"], dtype=torch.float32, device=device)
    pg_max_t = torch.tensor(case["pg_max"], dtype=torch.float32, device=device)
    r_max_t = torch.tensor(instances["r_max"], dtype=torch.float32, device=device)

    # Set lambda based on model and mode (following paper's Appendix B)
    if args.lam is not None:
        lam = args.lam
    else:
        if args.mode == "ssl":
            lam = 0.1
        else:
            lam = 1e-4
    mu = lam  # mu = lambda for SL (per paper)

    print(f"\nTraining: mode={args.mode.upper()}, lambda={lam}, mu={mu}")

    # -----------------------------------------------------------------------
    # 8. Train
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} with {args.mode.upper()}")
    print(f"{'='*60}")

    model, history = train_model(
        model, datasets, case, args.problem, mode=args.mode,
        lam=lam, mu=mu,
        b_branch_t=b_branch_t, branch_from=branch_from, branch_to=branch_to,
        branch_rate_t=branch_rate_t, B_bus_t=B_bus_t, gen_bus_idx_t=gen_bus_idx_t,
        cost_coef_t=cost_coef_t, pg_max_t=pg_max_t, r_max_t=r_max_t,
        lr=args.lr, weight_decay=1e-6,
        batch_size_train=args.batch_size, batch_size_eval=256,
        max_epochs=args.max_epochs, patience=args.patience,
        lr_patience=10, max_time_min=args.max_time,
        device=device, verbose=True,
    )

    # -----------------------------------------------------------------------
    # 9. Evaluate
    # -----------------------------------------------------------------------
    if obj_star is not None:
        print(f"\n{'='*60}")
        print("Evaluating on test set")
        print(f"{'='*60}")

        results = evaluate_model(
            model, datasets["test"], case, args.problem,
            b_branch_t, branch_from, branch_to, branch_rate_t,
            B_bus_t, gen_bus_idx_t,
            cost_coef_t, pg_max_t, r_max_t,
            batch_size=256, tol=1e-4, device=device,
        )

        print(f"\n--- Results ({args.model.upper()}, {args.mode.upper()}, {args.problem.upper()}) ---")
        print(f"  Mean Optimality Gap:     {results['mean_gap_pct']:.4f}%")
        print(f"  SGM Optimality Gap:      {results['sgm_gap_pct']:.4f}%")
        print(f"  Feasibility Rate:        {results['feasibility_rate_pct']:.2f}%")
        print(f"  Mean PB Violation:       {results['mean_pb_violation_pu']:.6f} p.u.")
        print(f"  Mean Reserve Shortage:   {results['mean_reserve_shortage_pu']:.6f} p.u.")
        print(f"  Mean DC Violation:       {results['mean_dc_violation_pu']:.6f} p.u.")
        print(f"  Training Time:           {history['training_time_min']:.1f} min")
    else:
        print("\nNo ground truth available — skipping evaluation metrics.")

    print("\nDone.")

