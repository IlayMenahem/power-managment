"""
OPF utilities: extract c_linear, f_min/f_max from gencost/branch; compute PTDF and Phi.

MATPOWER indices (0-based):
- branch: F_BUS=0, T_BUS=1, BR_R=2, BR_X=3, BR_B=4, RATE_A=5, TAP=8, SHIFT=9, BR_STATUS=10
- bus: TYPE=1 (1=PQ, 2=PV, 3=slack)
- gencost: MODEL=0, NCOST=4; polynomial coeffs cn..c1,c0 in columns 4,5,..., so c1 at col 4+n-2 for n>=2
"""

from __future__ import annotations

import numpy as np

# Branch columns (0-based)
BR_F_BUS = 0
BR_T_BUS = 1
BR_BR_R = 2
BR_BR_X = 3
BR_RATE_A = 5
BR_TAP = 8
BR_SHIFT = 9
BR_STATUS = 10

# Bus columns (0-based)
BUS_TYPE = 1
BUS_TYPE_REF = 3  # slack (reference)

# Gencost: model=0, startup=1, shutdown=2, ncost=3 (1-based 4), then c(n-1)..c0
# For polynomial order n, coeffs at 0-based indices 4..4+n (c_{n-1}, ..., c_0)
# Linear coefficient c1 is at index 4 + n - 2 = 4 + n - 2 (for n=2: index 4; for n=3: index 5)
GENCOST_MODEL = 0
GENCOST_NCOST = 3  # 0-based column for N (number of cost coeffs)


def extract_c_linear(
    gencost_matrix: np.ndarray,
    on_mask: np.ndarray,
) -> np.ndarray:
    """
    Extract linear cost coefficient per generator for online generators only.

    gencost_matrix: full (n_gen_total, n_cols) from mpc.gencost.
    on_mask: boolean (n_gen_total,) True for online generators (same order as gen_matrix).

    Returns:
        c_linear: (n_gen,) linear coefficient c1 for c(p) = sum_g c_g * p_g.
    """
    gencost_on = gencost_matrix[on_mask]
    n_gen = gencost_on.shape[0]
    c_linear = np.zeros(n_gen, dtype=np.float64)
    for g in range(n_gen):
        row = gencost_on[g]
        ncost = int(row[GENCOST_NCOST])
        if ncost < 2:
            c_linear[g] = 0.0
            continue
        # Polynomial coeffs start at 0-based index 4: c_{n-1}, c_{n-2}, ..., c_0
        # So c1 (coefficient of p^1) is at index 4 + (ncost - 1) - 1 = 4 + ncost - 2
        idx_c1 = 4 + ncost - 2
        if idx_c1 < row.shape[0]:
            c_linear[g] = float(row[idx_c1])
    return np.maximum(c_linear, 0.0)  # non-negative cost


def extract_f_min_f_max(
    branch_matrix: np.ndarray,
    base_mva: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract branch thermal limits from mpc.branch (in-service branches only).

    branch_matrix: (n_branches, n_cols) from mpc.branch.
    base_mva: base MVA for per-unit; use 1.0 to keep RATE_A in MW/MVA.

    Returns:
        f_min: (n_branches_in,) lower limit (typically -RATE_A).
        f_max: (n_branches_in,) upper limit (RATE_A).
    """
    if branch_matrix.shape[1] > BR_STATUS:
        in_service = branch_matrix[:, BR_STATUS] == 1
    else:
        in_service = np.ones(branch_matrix.shape[0], dtype=bool)
    branch = branch_matrix[in_service]
    rate_a = branch[:, BR_RATE_A].astype(np.float64)
    # RATE_A in MVA; for real power flow use as MW (same magnitude)
    # Replace zero/inf with large finite for numerical stability
    rate_a = np.where(rate_a <= 0, 1e6, rate_a)
    rate_a = np.where(np.isinf(rate_a), 1e6, rate_a)
    f_max = rate_a  # MW
    f_min = -rate_a
    return f_min, f_max


def _find_slack_bus(bus_matrix: np.ndarray) -> int:
    """Return 0-based index of slack bus (first bus with type 3)."""
    types = bus_matrix[:, BUS_TYPE].astype(int)
    slack_idx = np.flatnonzero(types == BUS_TYPE_REF)
    if slack_idx.size == 0:
        # Fallback: use first bus
        return 0
    return int(slack_idx[0])


def build_ptdf_and_phi(
    bus_matrix: np.ndarray,
    branch_matrix: np.ndarray,
    gen_bus: np.ndarray,
    base_mva: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build PTDF_bus (n_branch x n_bus) and Phi = PTDF_bus @ Cg (n_branch x n_gen).

    DC power flow: flow = B_f @ theta, B_bus @ theta = P_inj. Remove slack row/col
    to invert B_bus; then PTDF_bus maps P_inj (full n_bus) to flow.
    Cg is (n_bus x n_gen) with Cg[gen_bus[g], g] = 1.

    bus_matrix: (n_buses, n_cols) from mpc.bus; column 0 is bus number (bus_i).
    branch_matrix: (n_branches, n_cols); F_BUS, T_BUS are bus numbers (not row indices).
    gen_bus: (n_gen,) 0-based row index into bus_matrix for each (online) generator.
    base_mva: base MVA (for consistency; B matrix is in p.u.).

    Returns:
        PTDF_bus: (n_branches, n_buses) such that flow = PTDF_bus @ P_inj.
        Phi: (n_branches, n_gen) such that flow = Phi @ p - PTDF_bus @ d (for P_inj = Cg@p - d).
    """
    n_bus = bus_matrix.shape[0]
    # Map bus number (column 0) to 0-based row index
    bus_ids = bus_matrix[:, 0].astype(int)
    bus_id_to_idx = {int(bus_ids[i]): i for i in range(n_bus)}

    if branch_matrix.shape[1] > BR_STATUS:
        in_service = branch_matrix[:, BR_STATUS] == 1
    else:
        in_service = np.ones(branch_matrix.shape[0], dtype=bool)
    branch = branch_matrix[in_service]
    n_branch = branch.shape[0]
    n_gen = gen_bus.shape[0]

    slack_0 = _find_slack_bus(bus_matrix)
    non_slack = np.array([i for i in range(n_bus) if i != slack_0], dtype=np.intp)
    n_red = len(non_slack)

    # Build B_bus (n_bus x n_bus): B_ii += 1/x_ij, B_ij -= 1/x_ij for each branch (i,j)
    # F_BUS, T_BUS in branch are bus numbers -> map to 0-based indices
    B_bus = np.zeros((n_bus, n_bus))
    for k in range(n_branch):
        f_id = int(branch[k, BR_F_BUS])
        t_id = int(branch[k, BR_T_BUS])
        f = bus_id_to_idx.get(f_id, -1)
        t = bus_id_to_idx.get(t_id, -1)
        if f < 0 or t < 0:
            continue
        x = branch[k, BR_BR_X]
        tap = branch[k, BR_TAP] if branch.shape[1] > BR_TAP else 1.0
        if tap == 0:
            tap = 1.0
        b_ij = 1.0 / (x * tap)
        B_bus[f, f] += b_ij
        B_bus[t, t] += b_ij
        B_bus[f, t] -= b_ij
        B_bus[t, f] -= b_ij

    # Reduced B (drop slack row and column)
    B_red = np.delete(np.delete(B_bus, slack_0, 0), slack_0, 1)

    # B_f: n_branch x n_bus. Row k for branch (f,t): (1/x) at f, (-1/x) at t
    B_f = np.zeros((n_branch, n_bus))
    for k in range(n_branch):
        f_id = int(branch[k, BR_F_BUS])
        t_id = int(branch[k, BR_T_BUS])
        f = bus_id_to_idx.get(f_id, -1)
        t = bus_id_to_idx.get(t_id, -1)
        if f < 0 or t < 0:
            continue
        x = branch[k, BR_BR_X]
        tap = branch[k, BR_TAP] if branch.shape[1] > BR_TAP else 1.0
        if tap == 0:
            tap = 1.0
        b_ij = 1.0 / (x * tap)
        B_f[k, f] = b_ij
        B_f[k, t] = -b_ij

    # B_f_red: same but only non-slack columns (theta_red)
    B_f_red = B_f[:, non_slack]

    # theta_red = B_red^{-1} @ P_inj_red  =>  flow = B_f_red @ B_red^{-1} @ P_inj_red
    B_red_inv = np.linalg.inv(B_red)
    PTDF_red = B_f_red @ B_red_inv  # (n_branch x n_red)

    # Map P_inj (full n_bus) to P_inj_red (n_red): drop slack row
    M = np.zeros((n_red, n_bus))
    for i in range(n_red):
        M[i, non_slack[i]] = 1.0
    PTDF_bus = PTDF_red @ M  # (n_branch x n_bus)

    # Cg: (n_bus x n_gen), Cg[gen_bus[g], g] = 1 (gen_bus is already 0-based row index)
    Cg = np.zeros((n_bus, n_gen))
    for g in range(n_gen):
        b = int(gen_bus[g])
        if 0 <= b < n_bus:
            Cg[b, g] = 1.0

    Phi = PTDF_bus @ Cg  # (n_branch x n_gen)
    return PTDF_bus, Phi
