"""
Data utilities for E2ELR: MATPOWER parsing, PTDF computation,
instance generation, and LP solving via SciPy HiGHS.

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2)
"""

import re
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import linprog

from train import M_TH

# ---------------------------------------------------------------------------
# 1. MATPOWER Parser
# ---------------------------------------------------------------------------

def parse_matpower(filepath):
    """Parse a MATPOWER .m file and return a dict with bus, gen, gencost, branch data."""
    with open(filepath, "r") as f:
        text = f.read()

    baseMVA = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text).group(1))

    def _extract_matrix(name):
        pattern = rf"mpc\.{name}\s*=\s*\[(.*?)\]"
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            raise ValueError(f"Could not find mpc.{name} in {filepath}")
        raw = match.group(1)
        rows = []
        for line in raw.strip().split("\n"):
            line = line.split("%")[0].strip().rstrip(";").strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
        return np.array(rows)

    bus = _extract_matrix("bus")
    gen = _extract_matrix("gen")
    gencost = _extract_matrix("gencost")
    branch = _extract_matrix("branch")

    return {
        "baseMVA": baseMVA,
        "bus": bus,
        "gen": gen,
        "gencost": gencost,
        "branch": branch,
    }


def extract_case_data(raw):
    """
    Extract the relevant vectors/matrices from parsed MATPOWER data.
    All quantities are in per-unit (divided by baseMVA).

    Returns a dict with:
        bus_ids, n_bus, pd_ref, gen_bus_idx, gen_bus_ids, n_gen, pg_max,
        cost_coef, branch_from, branch_to, branch_x, branch_tap,
        branch_rate, n_branch, baseMVA
    """
    baseMVA = raw["baseMVA"]
    bus = raw["bus"]
    gen = raw["gen"]
    gencost = raw["gencost"]
    branch = raw["branch"]

    # Bus data
    bus_ids = bus[:, 0].astype(int)
    bus_id_to_idx = {bid: i for i, bid in enumerate(bus_ids)}
    n_bus = len(bus_ids)
    pd_ref = bus[:, 2] / baseMVA  # active power demand in p.u.

    # Generator data: keep only online generators with Pmax > 0
    gen_mask = (gen[:, 7] == 1) & (gen[:, 8] > 0)  # status=1, Pmax>0
    gen = gen[gen_mask]
    gencost = gencost[gen_mask]

    gen_bus_ids = gen[:, 0].astype(int)
    gen_bus_idx = np.array([bus_id_to_idx[b] for b in gen_bus_ids])
    n_gen = len(gen)
    pg_max = gen[:, 8] / baseMVA  # Pmax in p.u.

    # Cost coefficients: gencost format is [2, startup, shutdown, n, c(n-1), ..., c0]
    # With n=3: columns [4]=c2 (quad), [5]=c1 (linear), [6]=c0 (constant)
    # The paper uses linear cost only.
    n_coef = gencost[:, 3].astype(int)
    cost_coef = np.zeros(n_gen)
    for i in range(n_gen):
        # linear coefficient is the second-to-last: column index = 4 + n_coef[i] - 2
        cost_coef[i] = gencost[i, 4 + n_coef[i] - 2]

    # Branch data: keep only online branches
    branch_mask = branch[:, 10] == 1  # status=1
    branch = branch[branch_mask]
    n_branch = len(branch)

    branch_from = np.array([bus_id_to_idx[int(b)] for b in branch[:, 0]])
    branch_to = np.array([bus_id_to_idx[int(b)] for b in branch[:, 1]])
    branch_x = branch[:, 3]  # reactance
    branch_tap = branch[:, 8]  # tap ratio (0 means 1)
    branch_tap[branch_tap == 0] = 1.0
    branch_rate = branch[:, 5] / baseMVA  # thermal limit in p.u.

    return {
        "bus_ids": bus_ids,
        "bus_id_to_idx": bus_id_to_idx,
        "n_bus": n_bus,
        "pd_ref": pd_ref,
        "gen_bus_ids": gen_bus_ids,
        "gen_bus_idx": gen_bus_idx,
        "n_gen": n_gen,
        "pg_max": pg_max,
        "cost_coef": cost_coef,
        "branch_from": branch_from,
        "branch_to": branch_to,
        "branch_x": branch_x,
        "branch_tap": branch_tap,
        "branch_rate": branch_rate,
        "n_branch": n_branch,
        "baseMVA": baseMVA,
    }


# ---------------------------------------------------------------------------
# 2. B-Matrix Computation (replaces PTDF)
# ---------------------------------------------------------------------------

def compute_B_matrix(case):
    """
    Compute the DC power flow B-matrix and all derived quantities needed
    for neural-network training and the LP solver.

    Returns a tuple:
        B_bus         : sparse CSR (n_bus, n_bus) — full bus susceptance matrix
        B_pinv        : ndarray (n_bus, n_bus)    — pseudoinverse; slack row/col = 0
        b_branch      : ndarray (n_branch,)       — branch susceptances
        B_reduced_csc : sparse CSC (n_reduced, n_reduced) — for LP equality constraints
        F_theta       : ndarray (n_branch, n_reduced) — branch-angle sensitivity
        C_g_r         : ndarray (n_reduced, n_gen)    — gen-bus matrix at non-slack buses
        non_slack     : ndarray (n_reduced,) int      — non-slack bus indices
    """
    n_bus = case["n_bus"]
    n_branch = case["n_branch"]
    n_gen = case["n_gen"]
    branch_from = case["branch_from"]
    branch_to = case["branch_to"]
    branch_x = case["branch_x"]
    branch_tap = case["branch_tap"]
    gen_bus_idx = case["gen_bus_idx"]

    b_branch = 1.0 / (branch_x * branch_tap)

    # Build incidence matrix A (n_branch, n_bus) and susceptance diagonal B_diag
    row_idx = np.concatenate([np.arange(n_branch), np.arange(n_branch)])
    col_idx = np.concatenate([branch_from, branch_to])
    data = np.concatenate([np.ones(n_branch), -np.ones(n_branch)])
    A = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_branch, n_bus))
    B_diag = sparse.diags(b_branch)
    B_bus = A.T @ B_diag @ A  # (n_bus, n_bus) sparse

    # Identify slack bus (type-3 reference bus; default to bus 0)
    bus_types = case.get("bus_types", None)
    slack_idx = 0
    if bus_types is not None:
        ref_buses = np.where(bus_types == 3)[0]
        if len(ref_buses) > 0:
            slack_idx = ref_buses[0]

    keep = np.ones(n_bus, dtype=bool)
    keep[slack_idx] = False
    non_slack = np.where(keep)[0]
    n_reduced = len(non_slack)

    # Reduced B matrix for LP equality constraints
    B_reduced_csc = B_bus[np.ix_(keep, keep)].tocsc()

    # Pseudoinverse via sparse LU: solve B_reduced @ x = e_j for each column
    lu = splu(B_reduced_csc)
    B_inv_reduced = np.zeros((n_reduced, n_reduced))
    I_cols = np.eye(n_reduced)
    for j in range(n_reduced):
        B_inv_reduced[:, j] = lu.solve(I_cols[:, j])

    # Expand to full n_bus × n_bus (slack row/col remain zero)
    B_pinv = np.zeros((n_bus, n_bus))
    B_pinv[np.ix_(non_slack, non_slack)] = B_inv_reduced

    # Branch-angle sensitivity matrix: F_theta[e, j'] = (B_diag @ A)[e, non_slack[j']]
    # Used in LP thermal constraints: f = F_theta @ theta_r
    F_theta = np.asarray((B_diag @ A)[:, keep].todense())  # (n_branch, n_reduced)

    # Generator-bus connection matrix restricted to non-slack buses
    C_g_r = np.zeros((n_reduced, n_gen))
    for g in range(n_gen):
        bus = gen_bus_idx[g]
        if keep[bus]:
            r_idx = int(np.searchsorted(non_slack, bus))
            C_g_r[r_idx, g] = 1.0

    return B_bus, B_pinv, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack


# ---------------------------------------------------------------------------
# 3. Instance Generation
# ---------------------------------------------------------------------------

def compute_reserve_params(case):
    """Compute reserve parameters alpha_r and r_max as described in Section V-A."""
    pg_max = case["pg_max"]
    alpha_r = 5.0 * np.max(pg_max) / np.sum(pg_max)
    r_max = alpha_r * pg_max
    return alpha_r, r_max


def generate_instances(case, n_instances, problem_type="ed", seed=42):
    """
    Generate ED or ED-R instances by perturbing the reference load profile.

    Returns dict with:
        pd       : (n_instances, n_bus)   – demand vectors
        R_req    : (n_instances,)         – reserve requirement (0 for ED)
        r_max    : (n_gen,)               – max reserve per generator (0 for ED)
    """
    rng = np.random.RandomState(seed)
    pd_ref = case["pd_ref"]
    pg_max = case["pg_max"]
    n_bus = case["n_bus"]

    # Global scaling factor gamma ~ U[0.8, 1.2]
    gamma = rng.uniform(0.8, 1.2, size=n_instances)

    # Per-bus multiplicative noise eta ~ LogNormal(mean=1, std=0.05)
    # LogNormal with mu=0, sigma: mean = exp(mu + sigma^2/2) = 1 => mu = -sigma^2/2
    sigma_ln = 0.05
    mu_ln = -0.5 * sigma_ln ** 2
    eta = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=(n_instances, n_bus))

    pd = gamma[:, None] * eta * pd_ref[None, :]

    # Reserve parameters
    if problem_type == "edr":
        alpha_r, r_max = compute_reserve_params(case)
        # R ~ U[1, 2] * max(pg_max)
        R_req = rng.uniform(1.0, 2.0, size=n_instances) * np.max(pg_max)
    else:
        r_max = np.zeros_like(pg_max)
        R_req = np.zeros(n_instances)

    return {"pd": pd, "R_req": R_req, "r_max": r_max}


# ---------------------------------------------------------------------------
# 4. LP Solver — SciPy HiGHS (explicit DC power flow via B-matrix)
# ---------------------------------------------------------------------------

def solve_ed_highs(pd_vec, case, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack,
                   problem_type="ed", R_req=0.0, r_max=None, M_th=15.0):
    """
    Solve a single ED/ED-R instance using scipy.optimize.linprog with HiGHS.

    Uses explicit voltage-angle variables (theta_r) and DC power flow equality
    constraints instead of the PTDF-based formulation.

    Variables:
        ED:   [pg (n_gen), xi (n_branch), theta_r (n_reduced)]
        ED-R: [pg (n_gen), xi (n_branch), theta_r (n_reduced), r (n_gen)]

    Returns:
        pg_opt : (n_gen,) optimal dispatch, or None if infeasible
        obj    : optimal objective value, or None
    """
    n_gen = case["n_gen"]
    n_branch = case["n_branch"]
    pg_max = case["pg_max"]
    cost_coef = case["cost_coef"]
    branch_rate = case["branch_rate"]
    D = float(np.sum(pd_vec))
    n_reduced = len(non_slack)
    pd_r = pd_vec[non_slack]  # demands at non-slack buses

    # Dense B_reduced for equality constraints
    B_reduced_dense = B_reduced_csc.toarray()  # (n_reduced, n_reduced)

    if problem_type == "edr" and r_max is not None:
        # Variables: [pg (n_gen), xi (n_branch), theta_r (n_reduced), r (n_gen)]
        n_theta = n_gen + n_branch  # offset to theta_r block
        n_r = n_gen + n_branch + n_reduced  # offset to r block
        n_vars = n_gen + n_branch + n_reduced + n_gen
        c = np.zeros(n_vars)
        c[:n_gen] = cost_coef
        c[n_gen:n_gen + n_branch] = M_th

        # Equalities: (1) power balance, (2) DC power flow
        n_eq = 1 + n_reduced
        A_eq = np.zeros((n_eq, n_vars))
        b_eq = np.zeros(n_eq)

        # (1) sum(pg) = D
        A_eq[0, :n_gen] = 1.0
        b_eq[0] = D

        # (2) B_reduced @ theta_r - C_g_r @ pg = -pd_r
        A_eq[1:, :n_gen] = -C_g_r
        A_eq[1:, n_theta:n_theta + n_reduced] = B_reduced_dense
        b_eq[1:] = -pd_r

        # Inequalities:
        # (a) F_theta @ theta_r - xi <= rate  (upper thermal)
        # (b) -F_theta @ theta_r - xi <= rate (lower thermal)
        # (c) -sum(r) <= -R_req
        # (d) pg + r <= pg_max
        # (e) r <= r_max
        n_ineq = 2 * n_branch + 1 + n_gen + n_gen
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)
        row = 0

        A_ub[row:row + n_branch, n_gen:n_gen + n_branch] = -np.eye(n_branch)
        A_ub[row:row + n_branch, n_theta:n_theta + n_reduced] = F_theta
        b_ub[row:row + n_branch] = branch_rate
        row += n_branch

        A_ub[row:row + n_branch, n_gen:n_gen + n_branch] = -np.eye(n_branch)
        A_ub[row:row + n_branch, n_theta:n_theta + n_reduced] = -F_theta
        b_ub[row:row + n_branch] = branch_rate
        row += n_branch

        A_ub[row, n_r:] = -1.0
        b_ub[row] = -R_req
        row += 1

        A_ub[row:row + n_gen, :n_gen] = np.eye(n_gen)
        A_ub[row:row + n_gen, n_r:] = np.eye(n_gen)
        b_ub[row:row + n_gen] = pg_max
        row += n_gen

        A_ub[row:row + n_gen, n_r:] = np.eye(n_gen)
        b_ub[row:row + n_gen] = r_max

        bounds = (
            [(0, pg_max[i]) for i in range(n_gen)]
            + [(0, None) for _ in range(n_branch)]
            + [(None, None) for _ in range(n_reduced)]
            + [(0, r_max[i]) for i in range(n_gen)]
        )
    else:
        # Variables: [pg (n_gen), xi (n_branch), theta_r (n_reduced)]
        n_theta = n_gen + n_branch
        n_vars = n_gen + n_branch + n_reduced
        c = np.zeros(n_vars)
        c[:n_gen] = cost_coef
        c[n_gen:n_gen + n_branch] = M_th

        n_eq = 1 + n_reduced
        A_eq = np.zeros((n_eq, n_vars))
        b_eq = np.zeros(n_eq)

        A_eq[0, :n_gen] = 1.0
        b_eq[0] = D

        A_eq[1:, :n_gen] = -C_g_r
        A_eq[1:, n_theta:] = B_reduced_dense
        b_eq[1:] = -pd_r

        n_ineq = 2 * n_branch
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)

        A_ub[:n_branch, n_gen:n_gen + n_branch] = -np.eye(n_branch)
        A_ub[:n_branch, n_theta:] = F_theta
        b_ub[:n_branch] = branch_rate

        A_ub[n_branch:, n_gen:n_gen + n_branch] = -np.eye(n_branch)
        A_ub[n_branch:, n_theta:] = -F_theta
        b_ub[n_branch:] = branch_rate

        bounds = (
            [(0, pg_max[i]) for i in range(n_gen)]
            + [(0, None) for _ in range(n_branch)]
            + [(None, None) for _ in range(n_reduced)]
        )

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if result.success:
        return result.x[:n_gen], result.fun
    return None, None


# ---------------------------------------------------------------------------
# 5. Batch / Parallel Solving
# ---------------------------------------------------------------------------

def _solve_one_highs(args):
    """Worker function for multiprocessing (must be at module level)."""
    i, pd_vec, case, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack, \
        problem_type, R_req, r_max, M_th = args
    return solve_ed_highs(
        pd_vec, case, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack,
        problem_type=problem_type, R_req=R_req, r_max=r_max, M_th=M_th,
    )


def solve_all_instances(instances, case, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack,
                        problem_type="ed", M_th=15.0, verbose=True, n_workers=4):
    """
    Solve all instances in parallel using HiGHS and return optimal dispatches
    and objective values.

    Args:
        n_workers : number of parallel workers

    Returns:
        pg_star : (n_instances, n_gen) optimal dispatches
        obj_star: (n_instances,) optimal objective values
    """
    from multiprocessing import Pool

    pd_all = instances["pd"]
    R_all = instances["R_req"]
    r_max = instances["r_max"]
    n_instances = len(pd_all)
    n_gen = case["n_gen"]

    pg_star = np.zeros((n_instances, n_gen))
    obj_star = np.zeros(n_instances)
    n_failed = 0

    r_max_arg = r_max if problem_type == "edr" else None
    work_items = [
        (i, pd_all[i], case, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack,
         problem_type, R_all[i], r_max_arg, M_th)
        for i in range(n_instances)
    ]

    if verbose:
        print(f"  Solving {n_instances} instances with HiGHS "
              f"({n_workers} workers)...")

    with Pool(n_workers) as pool:
        results = pool.map(_solve_one_highs, work_items)

    for i, (pg_opt, obj) in enumerate(results):
        if pg_opt is not None:
            pg_star[i] = pg_opt
            obj_star[i] = obj
        else:
            n_failed += 1
            pg_star[i] = np.nan
            obj_star[i] = np.nan

    if verbose:
        print(f"  Done. {n_failed}/{n_instances} infeasible instances.")
    return pg_star, obj_star


# ---------------------------------------------------------------------------
# Main: generate dataset and save to checkpoint folder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import time

    parser = argparse.ArgumentParser(
        description="Generate ED/ED-R dataset and save ground truth to checkpoints."
    )
    parser.add_argument("--case", type=str, default="data/pglib_opf_case300_ieee.m",
                        help="Path to MATPOWER .m case file")
    parser.add_argument("--problem", type=str, default="ed", choices=["ed", "edr"],
                        help="Problem type: ed (no reserves) or edr (with reserves)")
    parser.add_argument("--n_instances", type=int, default=50000,
                        help="Total number of instances to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for saving ground truth solutions")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    # 1. Load case
    print(f"Loading case: {args.case}")
    raw = parse_matpower(args.case)
    case = extract_case_data(raw)
    print(f"  Buses: {case['n_bus']}, Generators: {case['n_gen']}, "
          f"Branches: {case['n_branch']}")

    # 2. Compute B-matrix
    print("Computing B-matrix...")
    t0 = time.time()
    B_bus, B_pinv, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack = compute_B_matrix(case)
    print(f"  B-matrix done in {time.time() - t0:.2f}s  "
          f"(B_pinv shape: {B_pinv.shape})")

    # 3. Generate instances
    print(f"Generating {args.n_instances} {args.problem.upper()} instances "
          f"(seed={args.seed})...")
    instances = generate_instances(case, args.n_instances, args.problem, seed=args.seed)

    if args.problem == "edr":
        alpha_r, r_max = compute_reserve_params(case)
        instances["r_max"] = r_max
        print(f"  alpha_r = {alpha_r:.4f}")

    # 4. Solve all instances
    print(f"Solving with HiGHS ({args.n_workers} workers)...")
    t0 = time.time()
    pg_star, obj_star = solve_all_instances(
        instances, case, b_branch, B_reduced_csc, F_theta, C_g_r, non_slack,
        problem_type=args.problem, M_th=M_TH,
        verbose=True, n_workers=args.n_workers,
    )
    elapsed = time.time() - t0
    print(f"  Solving done in {elapsed / 60:.1f} min")

    valid = ~np.isnan(obj_star)
    print(f"  Valid solutions: {valid.sum()}/{len(obj_star)}")
    if valid.any():
        print(f"  Objective range: [{obj_star[valid].min():.2f}, "
              f"{obj_star[valid].max():.2f}]")

    # 5. Save checkpoint
    case_name = os.path.splitext(os.path.basename(args.case))[0]
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"gt_{case_name}_{args.problem}_n{args.n_instances}_s{args.seed}.npz",
    )
    np.savez(checkpoint_path, pg_star=pg_star, obj_star=obj_star)
    print(f"  Saved to: {checkpoint_path}")
