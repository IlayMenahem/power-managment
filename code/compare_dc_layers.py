"""
Compare DC power flow repair layers inside E2ELR (ED-R, SSL).

Variants (all trained with the best E2ELRDC hyperparameters from ray_results/):
  - e2elrdc        : projection layer theta + B_pinv (pg_bus - pd - B theta),
                     B_pinv = torch.linalg.pinv(B) in float32 (as in E2ELRDCModel)
  - e2elrdc_pinv64 : same projection layer, B_pinv computed in float64 (ablation)
  - e2elrdc_ref    : reference-bus layer, B_r theta_r = (pg_bus - pd)_r (sparse LU)

Usage:
    python compare_dc_layers.py                # train all jobs, then summarize
    python compare_dc_layers.py --summarize    # only summarize saved results
"""

import argparse
import json
import os
import time
from itertools import product
from multiprocessing import get_context

import numpy as np
import torch
from data_utils import (
    compute_B_matrix,
    compute_reserve_params,
    extract_case_data,
    generate_instances,
    parse_matpower,
)
from models import E2ELRDCModel, E2ELRDCRefModel
from torch.utils.data import DataLoader
from train import (
    M_DC,
    M_TH,
    build_datasets,
    compute_thermal_violations,
    evaluate_model,
    train_model,
)

CASES = ["pglib_opf_case300_ieee", "pglib_opf_case1354_pegase"]
VARIANTS = ["e2elrdc", "e2elrdc_pinv64", "e2elrdc_ref"]
PROBLEM = "edr"
N_INSTANCES = 50000
DATA_SEED = 42
OUT_DIR = os.path.join("train_results", "dc_layer_comparison")


def load_problem(case_name):
    case = extract_case_data(parse_matpower(os.path.join("data", f"{case_name}.m")))
    B_bus, B_pinv, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack = (
        compute_B_matrix(case)
    )
    instances = generate_instances(case, N_INSTANCES, PROBLEM, seed=DATA_SEED)
    _, instances["r_max"] = compute_reserve_params(case)
    gt = np.load(
        os.path.join(
            "checkpoints", f"gt_{case_name}_{PROBLEM}_n{N_INSTANCES}_s{DATA_SEED}.npz"
        )
    )
    datasets = build_datasets(instances, gt["pg_star"], gt["obj_star"], case, PROBLEM)
    return case, B_bus, b_branch, non_slack, instances, datasets


def build_variant(variant, case, B_bus, non_slack, r_max, config):
    kwargs = dict(
        n_bus=case["n_bus"],
        n_gen=case["n_gen"],
        pg_max=case["pg_max"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        problem_type=PROBLEM,
        r_max=r_max,
        B=B_bus,
        gen_bus_idx=case["gen_bus_idx"],
    )
    if variant == "e2elrdc_ref":
        return E2ELRDCRefModel(**kwargs, non_slack=non_slack)
    model = E2ELRDCModel(**kwargs)
    if variant == "e2elrdc_pinv64":
        B_pinv64 = np.linalg.pinv(B_bus.toarray())
        model.DC_power_flow.B_pinv_t.copy_(torch.tensor(B_pinv64))
    return model


@torch.no_grad()
def penalty_breakdown(model, dataset, case, B_bus, b_branch, cost_coef_t):
    """Per-instance cost terms relative to the optimum, with the DC residual in float64."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    n_gen, n_bus = case["n_gen"], case["n_bus"]
    C_g = np.zeros((n_bus, n_gen))
    C_g[case["gen_bus_idx"], np.arange(n_gen)] = 1.0
    b_branch_t = torch.tensor(b_branch, dtype=torch.float32)
    branch_rate_t = torch.tensor(case["branch_rate"], dtype=torch.float32)

    terms = {"gen": [], "thermal": [], "dc64": [], "dc64_pu": [], "obj": []}
    model.eval()
    for pd_b, R_b, _, obj_b in loader:
        D_b = pd_b.sum(dim=-1, keepdim=True)
        pg, theta = model(pd_b, D_b, R_b)
        xi = compute_thermal_violations(
            theta, b_branch_t, case["branch_from"], case["branch_to"], branch_rate_t
        )
        net = pg.double().numpy() @ C_g.T - pd_b.double().numpy()
        dc = np.abs(B_bus @ theta.double().numpy().T - net.T).sum(axis=0)
        terms["gen"].append((cost_coef_t * pg).sum(-1).numpy())
        terms["thermal"].append(M_TH * xi.sum(-1).numpy())
        terms["dc64"].append(M_DC * dc)
        terms["dc64_pu"].append(dc)
        terms["obj"].append(obj_b.numpy())
    t = {k: np.concatenate(v).astype(np.float64) for k, v in terms.items()}
    valid = ~np.isnan(t["obj"])
    obj = t["obj"][valid]
    return {
        "gap_fp64_pct": 100.0
        * np.mean((t["gen"] + t["thermal"] + t["dc64"])[valid] / obj - 1.0),
        "gen_cost_gap_pct": 100.0 * np.mean(t["gen"][valid] / obj - 1.0),
        "thermal_pen_pct": 100.0 * np.mean(t["thermal"][valid] / obj),
        "dc_pen_fp64_pct": 100.0 * np.mean(t["dc64"][valid] / obj),
        "dc_violation_fp64_pu": float(np.mean(t["dc64_pu"])),
    }


def run_job(job):
    case_name, variant, seed, max_time = job
    torch.set_num_threads(2)
    case, B_bus, b_branch, non_slack, instances, datasets = load_problem(case_name)
    with open(
        os.path.join("ray_results", f"best_config_e2elrdc_{PROBLEM}_ssl_{case_name}.json")
    ) as f:
        config = json.load(f)["config"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_variant(variant, case, B_bus, non_slack, instances["r_max"], config)

    tensors = dict(
        b_branch_t=torch.tensor(b_branch, dtype=torch.float32),
        branch_from=case["branch_from"],
        branch_to=case["branch_to"],
        branch_rate_t=torch.tensor(case["branch_rate"], dtype=torch.float32),
        B_bus_t=torch.tensor(B_bus.toarray(), dtype=torch.float32),
        gen_bus_idx_t=torch.tensor(case["gen_bus_idx"], dtype=torch.long),
        cost_coef_t=torch.tensor(case["cost_coef"], dtype=torch.float32),
        pg_max_t=torch.tensor(case["pg_max"], dtype=torch.float32),
        r_max_t=torch.tensor(instances["r_max"], dtype=torch.float32),
    )
    model, history = train_model(
        model,
        datasets,
        case,
        PROBLEM,
        mode="ssl",
        lam=1.0,
        mu=1.0,
        **tensors,
        lr=config["lr"],
        batch_size_train=config["batch_size"],
        max_time_min=max_time,
        verbose=False,
    )
    results = evaluate_model(
        model, datasets["test"], case, PROBLEM, **tensors, mode="ssl", lam=1.0
    )
    results.update(
        penalty_breakdown(
            model, datasets["test"], case, B_bus, b_branch, tensors["cost_coef_t"]
        )
    )
    record = {
        "case": case_name,
        "variant": variant,
        "seed": seed,
        "config": config,
        "epochs": len(history["val_loss"]),
        "training_time_min": history["training_time_min"],
        "best_val_loss": history["best_val_loss"],
        "results": {k: float(v) for k, v in results.items()},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{case_name}_{variant}_s{seed}.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(
        f"[done] {case_name} {variant} seed={seed}: gap {results['mean_gap_pct']:.2f}% "
        f"dc {results['mean_dc_violation_pu']:.3g} pu ({record['epochs']} epochs)",
        flush=True,
    )
    return record


def time_layers(case_name, n_rep=50, batch_size=256):
    """Forward+backward time of the DC layer alone and its stored entries."""
    torch.set_num_threads(1)
    case, B_bus, b_branch, non_slack, instances, datasets = load_problem(case_name)
    pd = datasets["test"].tensors[0][:batch_size]
    pg = torch.tensor(case["pg_max"], dtype=torch.float32).expand(batch_size, -1)
    pg = (pg * pd.sum(-1, keepdim=True) / pg.sum(-1, keepdim=True)).clone()
    pg.requires_grad_(True)
    theta_nn = torch.randn(batch_size, case["n_bus"], requires_grad=True)
    config = {"hidden_dim": 128, "n_layers": 2}
    out = {}
    for variant in ["e2elrdc", "e2elrdc_ref"]:
        layer = build_variant(
            variant, case, B_bus, non_slack, instances["r_max"], config
        ).DC_power_flow
        if variant == "e2elrdc_ref":
            fn = lambda: layer(pg, pd)  # noqa: E731
            stored = layer.lu.L.nnz + layer.lu.U.nnz
        else:
            fn = lambda: layer(pg, pd, theta_nn)  # noqa: E731
            stored = layer.B_pinv_t.numel() + B_bus.nnz
        for _ in range(5):
            fn().sum().backward()
        t0 = time.perf_counter()
        for _ in range(n_rep):
            fn().sum().backward()
        out[variant] = {
            "ms_per_batch": 1e3 * (time.perf_counter() - t0) / n_rep,
            "stored_entries": int(stored),
        }
    return out


def summarize():
    records = []
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn.endswith(".json") and not fn.startswith("summary"):
            with open(os.path.join(OUT_DIR, fn)) as f:
                records.append(json.load(f))
    cols = [
        ("mean_gap_pct", "Gap (%)"),
        ("sgm_gap_pct", "SGM Gap (%)"),
        ("mean_dc_violation_pu", "DC viol. fp32 (pu)"),
        ("dc_violation_fp64_pu", "DC viol. fp64 (pu)"),
        ("dc_pen_fp64_pct", "DC pen. (% of opt)"),
        ("thermal_pen_pct", "Thermal pen. (% of opt)"),
        ("gen_cost_gap_pct", "Gen. cost gap (%)"),
    ]
    summary = {}
    for case_name in CASES:
        print(f"\n### {case_name}  (mean ± std over seeds)\n")
        print("| Variant | seeds | epochs | " + " | ".join(c[1] for c in cols) + " |")
        print("|---" * (len(cols) + 3) + "|")
        for variant in VARIANTS:
            rs = [
                r for r in records if r["case"] == case_name and r["variant"] == variant
            ]
            if not rs:
                continue
            row = {}
            cells = []
            for key, _ in cols:
                v = np.array([r["results"][key] for r in rs])
                row[key] = (float(v.mean()), float(v.std()))
                cells.append(f"{v.mean():.3g} ± {v.std():.2g}")
            epochs = np.mean([r["epochs"] for r in rs])
            summary[f"{case_name}/{variant}"] = row
            print(f"| {variant} | {len(rs)} | {epochs:.0f} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max_time", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.timing:
        for case_name in args.cases:
            print(case_name, json.dumps(time_layers(case_name), indent=2))
    elif not args.summarize:
        jobs = list(product(args.cases, args.variants, args.seeds, [args.max_time]))
        with get_context("spawn").Pool(args.workers) as pool:
            pool.map(run_job, jobs, chunksize=1)
    if not args.timing:
        summarize()
