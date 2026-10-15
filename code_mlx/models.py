"""
Neural network optimization proxies for Economic Dispatch (MLX version).

Every architecture maps (pd, R) -> (pg, theta):
  - dnn   : vanilla DNN (Fig. 1a); pg = pg_min + sigmoid(DNN(pd, R)) * (pg_max - pg_min).
  - e2elr : DNN + power balance repair + reserve repair (Fig. 4); pg satisfies
            bounds, power balance and reserves.

The network input is the instance data that varies between instances, (pd, R)
(Section V: x = (c, d, R, ...); the other entries of x are the same for every
instance). The network predicts only pg. The angles theta are not predicted:
they solve the DC power flow equations for pg (theta_ref = 0,
B_r theta_r = (pg_bus - pd)_r), so the branch flows equal the paper's PTDF
flows Phi (pg - pd).

The backbone runs in float32 on the GPU. The repair layers, the DC power flow
and the returned (pg, theta) are float64, because float32 cannot represent
theta precisely enough for B @ theta = pg_bus - pd to hold within the
feasibility tolerance (B has entries up to ~2e4 p.u.). MLX supports float64
only on the CPU, so those parts run on the CPU stream (memory is unified, so
there is no copy between the two).

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2), Sections III-IV.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.sparse.linalg import splu
from data_utils import Case, bus_susceptance

# Autograd builds backward ops on the default device, so it must be the CPU
# (float64 is unsupported on the GPU); the backbone is explicitly put on the GPU.
mx.set_default_device(mx.cpu)

ARCHITECTURES = ("dnn", "e2elr")
EPS = 1e-8


def f64(x: np.ndarray) -> mx.array:
    """float64 array; every float64 op must run on the CPU stream."""
    with mx.stream(mx.cpu):
        return mx.array(np.asarray(x, dtype=np.float64), dtype=mx.float64)


def bus_injection(pg: mx.array, pd: mx.array, gen_bus: mx.array) -> mx.array:
    """Net injection pg_bus - pd at each bus: (batch, n_bus)."""
    return mx.zeros_like(pd).at[:, gen_bus].add(pg) - pd


# ---------------------------------------------------------------------------
# Repair layers (differentiable, closed-form)
# ---------------------------------------------------------------------------


def power_balance_repair(pg: mx.array, pg_max: mx.array, D: mx.array) -> mx.array:
    """
    Power balance repair (Eq. 6-7): given 0 <= pg <= pg_max, return a dispatch
    with sum(pg) = D that stays within bounds. A shortage moves every generator
    a common fraction of its headroom toward pg_max; a surplus moves it toward 0.

    Shapes: pg (batch, n_gen), pg_max (n_gen,), D (batch, 1).
    """
    total = pg.sum(axis=-1, keepdims=True)
    eta_up = mx.clip((D - total) / (pg_max.sum() - total + EPS), 0.0, 1.0)
    eta_down = mx.clip((total - D) / (total + EPS), 0.0, 1.0)
    pg_up = (1 - eta_up) * pg + eta_up * pg_max
    pg_down = (1 - eta_down) * pg
    return mx.where(total < D, pg_up, pg_down)


def reserve_repair(
    pg: mx.array, pg_max: mx.array, r_max: mx.array, R: mx.array
) -> mx.array:
    """
    Reserve repair (Algorithm 1): given pg satisfying bounds and power balance,
    shift dispatch toward pg_max - r_max so that reserve R can be allocated.
    Increases and decreases are equal, so power balance is preserved. With
    r_max = 0 and R = 0 (ED) this is the identity.

    Shapes: pg (batch, n_gen), pg_max and r_max (n_gen,), R (batch, 1).
    """
    r_star = mx.minimum(r_max, pg_max - pg)
    delta_R = R - r_star.sum(axis=-1, keepdims=True)  # reserve shortage

    target = pg_max - r_max  # dispatch at which the full reserve is available
    below = pg <= target  # G_up: can increase dispatch without losing reserve
    Delta_up = mx.where(below, target - pg, 0.0).sum(axis=-1, keepdims=True)
    Delta_down = mx.where(below, 0.0, pg - target).sum(axis=-1, keepdims=True)

    Delta = mx.maximum(mx.minimum(mx.minimum(delta_R, Delta_up), Delta_down), 0.0)
    alpha_up = Delta / (Delta_up + EPS)
    alpha_down = Delta / (Delta_down + EPS)

    pg_adj_up = (1 - alpha_up) * pg + alpha_up * target
    pg_adj_down = (1 - alpha_down) * pg + alpha_down * target
    return mx.where(below, pg_adj_up, pg_adj_down)


class DCPowerFlow(nn.Module):
    """
    Angles of a dispatch: theta_ref = 0 and B_r theta_r = (pg_bus - pd)_r,
    where B_r is B without the reference bus. B_r is nonsingular for a
    connected network and is factored once
    with a sparse LU (SciPy, float64); the forward pass is a triangular solve
    and the backward pass a solve with B_r^T (= B_r, as B is symmetric). The
    dropped reference-bus equation holds whenever the power balance holds.
    """

    def __init__(self, case: Case):
        super().__init__()
        non_slack = np.flatnonzero(np.arange(case.n_bus) != case.slack)
        self.lu = splu(bus_susceptance(case)[non_slack][:, non_slack].tocsc())
        self.n_bus = case.n_bus
        self._non_slack = mx.array(non_slack.astype(np.int32))
        self._gen_bus = mx.array(case.gen_bus.astype(np.int32))

        @mx.custom_function
        def solve(rhs_r: mx.array) -> mx.array:
            return self._solve(rhs_r, trans="N")

        @solve.vjp
        def solve_vjp(primals, cotangent, output):
            return self._solve(cotangent, trans="T")

        self._solve_fn = solve

    def _solve(self, rhs: mx.array, trans: str) -> mx.array:
        with mx.stream(mx.cpu):
            x = self.lu.solve(np.array(rhs).T, trans=trans).T
            return mx.array(np.ascontiguousarray(x), dtype=mx.float64)

    def __call__(self, pg: mx.array, pd: mx.array) -> mx.array:
        rhs = bus_injection(pg, pd, self._gen_bus)
        theta_r = self._solve_fn(rhs[:, self._non_slack])
        return mx.zeros((pg.shape[0], self.n_bus), dtype=mx.float64).at[
            :, self._non_slack
        ].add(theta_r)


# ---------------------------------------------------------------------------
# Proxy model
# ---------------------------------------------------------------------------


class DNNBackbone(nn.Module):
    """
    Fully-connected network with ReLU, BatchNorm and Dropout.
    Maps x = (pd, R) (batch, n_bus + 1) to z in [0, 1]^n_gen (sigmoid).
    """

    def __init__(
        self,
        n_bus: int,
        n_gen: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_bus + 1
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.pg_head = nn.Linear(hidden_dim, n_gen)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sigmoid(self.pg_head(self.shared(x)))


class DispatchProxy(nn.Module):
    """
    Optimization proxy for one of ARCHITECTURES; see the module docstring.

    The repair layers assume a minimum output of zero. As in the paper (Section
    III-A), nonzero minimums are handled by shifting: the layers act on
    pg - pg_min with limits pg_max - pg_min, demand D - sum(pg_min) and reserve
    capacity min(r_max, pg_max - pg_min).
    """

    def __init__(
        self, arch: str, case: Case, r_max: np.ndarray, hidden_dim: int, n_layers: int
    ):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"Unknown model: {arch}")
        self.backbone = DNNBackbone(case.n_bus, case.n_gen, hidden_dim, n_layers)
        pg_range = case.pg_max - case.pg_min
        self._pg_min = f64(case.pg_min)
        self._pg_range = f64(pg_range)
        self._r_max_shifted = f64(np.minimum(r_max, pg_range))
        self._repair = arch != "dnn"
        self.dc_power_flow = DCPowerFlow(case)

    def initial_dispatch(self, pd: mx.array, R: mx.array) -> mx.array:
        """Backbone prediction pg_min + z * (pg_max - pg_min), before any repair."""
        x = mx.concatenate([pd, R], axis=-1).astype(mx.float32)  # on the CPU
        with mx.stream(mx.gpu):  # float32 backbone
            z = self.backbone(x)
        with mx.stream(mx.cpu):  # float64 is CPU-only in MLX
            return self._pg_min + z.astype(mx.float64) * self._pg_range

    def repair(self, pg: mx.array, pd: mx.array, R: mx.array) -> mx.array:
        """Repair layers (E2ELR) on pg in [pg_min, pg_max]; the identity for dnn."""
        if not self._repair:
            return pg
        with mx.stream(mx.cpu):
            D = pd.sum(axis=-1, keepdims=True) - self._pg_min.sum()
            pg = power_balance_repair(pg - self._pg_min, self._pg_range, D)
            pg = reserve_repair(pg, self._pg_range, self._r_max_shifted, R)
            return pg + self._pg_min

    def dispatch(self, pd: mx.array, R: mx.array) -> mx.array:
        """The proxy's output p_hat (batch, n_gen)."""
        return self.repair(self.initial_dispatch(pd, R), pd, R)

    def __call__(self, pd: mx.array, R: mx.array) -> tuple[mx.array, mx.array]:
        """
        Args:
            pd : (batch, n_bus) float64 demand
            R  : (batch, 1)     float64 reserve requirement
        Returns:
            pg (batch, n_gen) and theta (batch, n_bus), both float64
        """
        pg = self.dispatch(pd, R)
        with mx.stream(mx.cpu):
            return pg, self.dc_power_flow(pg, pd)
