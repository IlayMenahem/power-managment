"""
Data utilities for E2ELR: MATPOWER parsing, DC network matrices, instance
generation, and ground-truth LP solving via SciPy HiGHS.

All power quantities are in per-unit (divided by baseMVA). ED is treated as
ED-R with zero reserve requirement and zero reserve capacity, so a single
formulation serves both problems.

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2)

Usage (generate and save the ground truth used by main.py):
    python data_utils.py --case data/pglib_opf_case300_ieee.m --problem edr
"""

import argparse
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

# MISO-based penalty prices (Section V-C), shared by the LP and the training losses.
M_TH = 1500.0  # thermal violation: 1500 $/MW
M_PB = 3500.0  # power balance: 3500 $/MW
M_RES = 1100.0  # reserve shortage: 1100 $/MW
M_DC = 3500.0  # DC power flow residual: 3500 $/MW


# ---------------------------------------------------------------------------
# 1. Case data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """DC network of a MATPOWER case, restricted to online generators and branches."""

    name: str
    base_mva: float
    slack: int  # index of the reference bus
    pd_ref: np.ndarray  # (n_bus,) reference active demand
    gen_bus: np.ndarray  # (n_gen,) bus index of each generator
    pg_max: np.ndarray  # (n_gen,) maximum generation
    cost: np.ndarray  # (n_gen,) linear cost coefficient
    branch_from: np.ndarray  # (n_branch,) from-bus index
    branch_to: np.ndarray  # (n_branch,) to-bus index
    b_branch: np.ndarray  # (n_branch,) susceptance 1 / (x * tap)
    rate: np.ndarray  # (n_branch,) thermal limit

    @property
    def n_bus(self) -> int:
        return len(self.pd_ref)

    @property
    def n_gen(self) -> int:
        return len(self.pg_max)

    @property
    def n_branch(self) -> int:
        return len(self.rate)

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_bus} buses, {self.n_gen} generators, "
            f"{self.n_branch} branches"
        )


def _matpower_matrix(text: str, name: str) -> np.ndarray:
    """Read the numeric matrix assigned to mpc.<name> in a MATPOWER file."""
    match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find mpc.{name}")
    rows = [line.split("%")[0].strip().rstrip(";") for line in match[1].splitlines()]
    return np.array([[float(x) for x in row.split()] for row in rows if row.strip()])


def load_case(path: str) -> Case:
    """Parse a MATPOWER .m file into a per-unit Case."""
    with open(path) as f:
        text = f.read()
    base_mva = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text)[1])
    bus, gen, gencost, branch = (
        _matpower_matrix(text, name) for name in ("bus", "gen", "gencost", "branch")
    )

    bus_index = {int(bus_id): i for i, bus_id in enumerate(bus[:, 0])}

    def to_bus_index(bus_ids: np.ndarray) -> np.ndarray:
        return np.array([bus_index[int(b)] for b in bus_ids])

    online_gen = (gen[:, 7] == 1) & (gen[:, 8] > 0)  # status = 1 and Pmax > 0
    gen, gencost = gen[online_gen], gencost[online_gen]
    # Polynomial gencost rows are [2, startup, shutdown, n, c_{n-1}, ..., c_0].
    # The paper uses the linear coefficient c_1 only, at column 4 + n - 2.
    cost = gencost[np.arange(len(gencost)), 2 + gencost[:, 3].astype(int)]

    branch = branch[branch[:, 10] == 1]  # status = 1
    tap = np.where(branch[:, 8] == 0, 1.0, branch[:, 8])  # tap ratio 0 means 1

    ref_buses = np.flatnonzero(bus[:, 1] == 3)
    return Case(
        name=os.path.splitext(os.path.basename(path))[0],
        base_mva=base_mva,
        slack=int(ref_buses[0]) if len(ref_buses) else 0,
        pd_ref=bus[:, 2] / base_mva,
        gen_bus=to_bus_index(gen[:, 0]),
        pg_max=gen[:, 8] / base_mva,
        cost=cost,
        branch_from=to_bus_index(branch[:, 0]),
        branch_to=to_bus_index(branch[:, 1]),
        b_branch=1.0 / (branch[:, 3] * tap),
        rate=branch[:, 5] / base_mva,
    )


def incidence_matrix(case: Case) -> sparse.csr_matrix:
    """Branch-bus incidence A (n_branch, n_bus): +1 at from-bus, -1 at to-bus."""
    rows = np.tile(np.arange(case.n_branch), 2)
    cols = np.concatenate([case.branch_from, case.branch_to])
    signs = np.repeat([1.0, -1.0], case.n_branch)
    return sparse.csr_matrix((signs, (rows, cols)), shape=(case.n_branch, case.n_bus))


def bus_susceptance(case: Case) -> sparse.csr_matrix:
    """DC power flow matrix B = A^T diag(b) A, so that B @ theta = pg_bus - pd."""
    A = incidence_matrix(case)
    return (A.T @ sparse.diags(case.b_branch) @ A).tocsr()


# ---------------------------------------------------------------------------
# 2. Instance generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instances:
    pd: np.ndarray  # (n_instances, n_bus) demand
    R: np.ndarray  # (n_instances,) reserve requirement, 0 for ED
    r_max: np.ndarray  # (n_gen,) reserve capacity per generator, 0 for ED


def compute_reserve_params(case: Case) -> tuple[float, np.ndarray]:
    """Reserve capacity r_max = alpha_r * pg_max, sum(r_max) = 5 max(pg_max) (V-A)."""
    alpha_r = 5.0 * np.max(case.pg_max) / np.sum(case.pg_max)
    return alpha_r, alpha_r * case.pg_max


def generate_instances(
    case: Case, n_instances: int, problem: str, seed: int = 42
) -> Instances:
    """Perturb the reference load: pd = gamma * eta * pd_ref (Section V-A)."""
    rng = np.random.RandomState(seed)
    gamma = rng.uniform(0.8, 1.2, size=n_instances)  # global scaling
    # Per-bus LogNormal noise with mean 1 and std ~0.05: mu = -sigma^2 / 2
    sigma = 0.05
    eta = rng.lognormal(
        mean=-0.5 * sigma**2, sigma=sigma, size=(n_instances, case.n_bus)
    )
    pd = gamma[:, None] * eta * case.pd_ref[None, :]

    if problem == "ed":
        return Instances(pd, np.zeros(n_instances), np.zeros(case.n_gen))
    R = rng.uniform(1.0, 2.0, size=n_instances) * np.max(case.pg_max)
    return Instances(pd, R, compute_reserve_params(case)[1])


# ---------------------------------------------------------------------------
# 3. LP solver: SciPy HiGHS with explicit DC power flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchLP:
    """
    Instance-independent part of the ED-R LP over x = [pg, xi, theta_r, r]:

        min   cost @ pg + M_TH * sum(xi)
        s.t.  sum(pg) = D                         power balance
              B_r @ theta_r - C_g @ pg = -pd_r    DC power flow at non-slack buses
              |F @ theta_r| - xi <= rate          soft thermal limits
              pg + r <= pg_max,  sum(r) >= R      reserves
              0 <= pg <= pg_max, xi >= 0, 0 <= r <= r_max
    """

    c: np.ndarray
    A_eq: sparse.csr_matrix
    A_ub: sparse.csr_matrix
    b_ub_fixed: np.ndarray  # b_ub without the last (reserve requirement) row
    bounds: np.ndarray  # (n_vars, 2)
    non_slack: np.ndarray  # (n_bus - 1,) non-slack bus indices
    n_gen: int


def build_lp(case: Case, r_max: np.ndarray) -> DispatchLP:
    n_gen, n_branch, n_bus = case.n_gen, case.n_branch, case.n_bus
    non_slack = np.flatnonzero(np.arange(n_bus) != case.slack)

    B_r = bus_susceptance(case)[non_slack][:, non_slack]
    F = (sparse.diags(case.b_branch) @ incidence_matrix(case))[:, non_slack]
    C_g = sparse.csr_matrix(
        (np.ones(n_gen), (case.gen_bus, np.arange(n_gen))), shape=(n_bus, n_gen)
    )[non_slack]
    ones_g = sparse.csr_matrix(np.ones((1, n_gen)))
    I_g, I_e = sparse.identity(n_gen), sparse.identity(n_branch)

    # Columns: pg | xi | theta_r | r.  The first n_bus rows are equalities.
    constraints = sparse.bmat(
        [
            [ones_g, None, None, None],  # sum(pg) = D
            [-C_g, None, B_r, None],  # B_r theta_r - C_g pg = -pd_r
            [None, -I_e, F, None],  # f - xi <= rate
            [None, -I_e, -F, None],  # -f - xi <= rate
            [I_g, None, None, I_g],  # pg + r <= pg_max
            [None, None, None, -ones_g],  # -sum(r) <= -R
        ],
        format="csr",
    )

    zeros_g, zeros_e, zeros_t = np.zeros(n_gen), np.zeros(n_branch), np.zeros(n_bus - 1)
    inf_e, inf_t = np.full(n_branch, np.inf), np.full(n_bus - 1, np.inf)
    return DispatchLP(
        c=np.concatenate([case.cost, np.full(n_branch, M_TH), zeros_t, zeros_g]),
        A_eq=constraints[:n_bus],
        A_ub=constraints[n_bus:],
        b_ub_fixed=np.concatenate([case.rate, case.rate, case.pg_max]),
        bounds=np.column_stack(
            [
                np.concatenate([zeros_g, zeros_e, -inf_t, zeros_g]),
                np.concatenate([case.pg_max, inf_e, inf_t, r_max]),
            ]
        ),
        non_slack=non_slack,
        n_gen=n_gen,
    )


def solve_instance(
    lp: DispatchLP, pd: np.ndarray, R: float
) -> tuple[np.ndarray, float]:
    """Return the optimal (pg, objective), or NaNs if HiGHS fails."""
    result = linprog(
        lp.c,
        A_ub=lp.A_ub,
        b_ub=np.append(lp.b_ub_fixed, -R),
        A_eq=lp.A_eq,
        b_eq=np.concatenate([[pd.sum()], -pd[lp.non_slack]]),
        bounds=lp.bounds,
        method="highs",
    )
    if not result.success:
        return np.full(lp.n_gen, np.nan), np.nan
    return result.x[: lp.n_gen], result.fun


def solve_all_instances(
    case: Case, instances: Instances, n_workers: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Solve all instances in parallel; returns pg_star (n, n_gen), obj_star (n,)."""
    solve = partial(solve_instance, build_lp(case, instances.r_max))
    print(
        f"  Solving {len(instances.pd)} instances with HiGHS ({n_workers} workers)..."
    )
    with Pool(n_workers) as pool:
        results = pool.starmap(
            solve, zip(instances.pd, instances.R, strict=True), chunksize=64
        )
    pg_star = np.stack([pg for pg, _ in results])
    obj_star = np.array([obj for _, obj in results])
    print(f"  {np.isnan(obj_star).sum()}/{len(obj_star)} instances failed (NaN).")
    return pg_star, obj_star


# ---------------------------------------------------------------------------
# 4. Ground-truth checkpoints
# ---------------------------------------------------------------------------


def ground_truth_path(
    checkpoint_dir: str, case: Case, problem: str, n_instances: int, seed: int
) -> str:
    filename = f"gt_{case.name}_{problem}_n{n_instances}_s{seed}.npz"
    return os.path.join(checkpoint_dir, filename)


def solve_and_save(
    case: Case, instances: Instances, path: str, n_workers: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    pg_star, obj_star = solve_all_instances(case, instances, n_workers)
    print(f"  Solving done in {(time.time() - t0) / 60:.1f} min")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, pg_star=pg_star, obj_star=obj_star)
    print(f"  Ground truth saved to: {path}")
    return pg_star, obj_star


def load_or_solve_ground_truth(
    case: Case, instances: Instances, path: str, solve_if_missing: bool
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the ground-truth checkpoint; solve it only if missing and requested."""
    if os.path.isfile(path):
        print(f"Loading ground truth from: {path}")
        data = np.load(path)
        return data["pg_star"], data["obj_star"]
    if solve_if_missing:
        return solve_and_save(case, instances, path)
    print("No ground truth checkpoint found; optimality gaps will be unavailable.")
    return None


# ---------------------------------------------------------------------------
# Main: generate dataset and save ground truth to the checkpoint folder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ED/ED-R instances and save ground truth to checkpoints."
    )
    parser.add_argument("--case", type=str, default="data/pglib_opf_case300_ieee.m")
    parser.add_argument("--problem", type=str, default="ed", choices=["ed", "edr"])
    parser.add_argument("--n_instances", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    case = load_case(args.case)
    print(f"Case {case.summary()}")
    instances = generate_instances(case, args.n_instances, args.problem, args.seed)
    path = ground_truth_path(
        args.checkpoint_dir, case, args.problem, args.n_instances, args.seed
    )
    _, obj_star = solve_and_save(case, instances, path, args.n_workers)
    valid = obj_star[~np.isnan(obj_star)]
    if len(valid):
        print(f"  Objective range: [{valid.min():.2f}, {valid.max():.2f}]")
