"""
Neural network optimization proxies for Economic Dispatch.

Every architecture maps (pd, R) -> (pg, theta):
  - dnn     : vanilla DNN (Fig. 2a); pg = sigmoid(DNN(pd)) * pg_max, theta predicted.
  - e2elr   : DNN + power balance repair + reserve repair (Fig. 3); pg satisfies
              bounds, power balance and reserves; theta predicted.
  - e2elrdc : E2ELR + projection of theta onto the DC power flow equations
              (proposed extension).

The backbone runs in float32. The repair layers, the DC projection and the
returned (pg, theta) are float64, because float32 cannot represent theta
precisely enough for B @ theta = pg_bus - pd to hold within the feasibility
tolerance (B has entries up to ~2e4 p.u.).

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2), Sections III-IV.
"""

import numpy as np
import torch
import torch.nn as nn
from data_utils import Case, bus_susceptance

ARCHITECTURES = ("dnn", "e2elr", "e2elrdc")
EPS = 1e-8


def bus_injection(
    pg: torch.Tensor, pd: torch.Tensor, gen_bus: torch.Tensor
) -> torch.Tensor:
    """Net injection pg_bus - pd at each bus: (batch, n_bus)."""
    return (-pd).index_add(1, gen_bus, pg)


# ---------------------------------------------------------------------------
# Repair layers (differentiable, closed-form)
# ---------------------------------------------------------------------------


def power_balance_repair(
    pg: torch.Tensor, pg_max: torch.Tensor, D: torch.Tensor
) -> torch.Tensor:
    """
    Power balance repair (Eq. 6-7): given 0 <= pg <= pg_max, return a dispatch
    with sum(pg) = D that stays within bounds. A shortage moves every generator
    a common fraction of its headroom toward pg_max; a surplus moves it toward 0.

    Shapes: pg (batch, n_gen), pg_max (n_gen,), D (batch, 1).
    """
    total = pg.sum(dim=-1, keepdim=True)
    eta_up = ((D - total) / (pg_max.sum() - total + EPS)).clamp(0.0, 1.0)
    eta_down = ((total - D) / (total + EPS)).clamp(0.0, 1.0)
    pg_up = (1 - eta_up) * pg + eta_up * pg_max
    pg_down = (1 - eta_down) * pg
    return torch.where(total < D, pg_up, pg_down)


def reserve_repair(
    pg: torch.Tensor, pg_max: torch.Tensor, r_max: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """
    Reserve repair (Algorithm 1): given pg satisfying bounds and power balance,
    shift dispatch toward pg_max - r_max so that reserve R can be allocated.
    Increases and decreases are equal, so power balance is preserved. With
    r_max = 0 and R = 0 (ED) this is the identity.

    Shapes: pg (batch, n_gen), pg_max and r_max (n_gen,), R (batch, 1).
    """
    r_star = torch.min(r_max, pg_max - pg)
    delta_R = R - r_star.sum(dim=-1, keepdim=True)  # reserve shortage

    target = pg_max - r_max  # dispatch at which the full reserve is available
    below = pg <= target  # G_up: can increase dispatch without losing reserve
    Delta_up = torch.where(below, target - pg, 0.0).sum(dim=-1, keepdim=True)
    Delta_down = torch.where(below, 0.0, pg - target).sum(dim=-1, keepdim=True)

    Delta = torch.min(torch.min(delta_R, Delta_up), Delta_down).clamp(min=0.0)
    alpha_up = Delta / (Delta_up + EPS)
    alpha_down = Delta / (Delta_down + EPS)

    pg_adj_up = (1 - alpha_up) * pg + alpha_up * target
    pg_adj_down = (1 - alpha_down) * pg + alpha_down * target
    return torch.where(below, pg_adj_up, pg_adj_down)


class DCProjection(nn.Module):
    """
    Closest theta to the backbone output satisfying B @ theta = pg_bus - pd:

        theta + B^+ (pg_bus - pd - B theta) = (I - B^+ B) theta + B^+ (pg_bus - pd)

    The right-hand form avoids the cancellation in B @ theta. For a connected
    network I - B^+ B averages theta, so only the mean of the predicted theta
    survives; it does not change any branch flow.
    """

    def __init__(self, case: Case):
        super().__init__()
        B = bus_susceptance(case).toarray()
        B_pinv = np.linalg.pinv(B)
        self.register_buffer("B_pinv", torch.tensor(B_pinv))
        self.register_buffer("null_proj", torch.tensor(np.eye(case.n_bus) - B_pinv @ B))
        self.register_buffer("gen_bus", torch.tensor(case.gen_bus))

    def forward(
        self, pg: torch.Tensor, pd: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        net = bus_injection(pg, pd, self.gen_bus)
        return theta @ self.null_proj.T + net @ self.B_pinv.T


# ---------------------------------------------------------------------------
# Proxy model
# ---------------------------------------------------------------------------


class DNNBackbone(nn.Module):
    """
    Fully-connected network with ReLU, BatchNorm and Dropout.
    Maps pd (batch, n_bus) to z in [0, 1]^n_gen (sigmoid) and unconstrained theta.
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
        in_dim = n_bus
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.pg_head = nn.Sequential(nn.Linear(hidden_dim, n_gen), nn.Sigmoid())
        self.theta_head = nn.Linear(hidden_dim, n_bus)

    def forward(self, pd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(pd)
        return self.pg_head(h), self.theta_head(h)


class DispatchProxy(nn.Module):
    """Optimization proxy for one of ARCHITECTURES; see the module docstring."""

    def __init__(
        self, arch: str, case: Case, r_max: np.ndarray, hidden_dim: int, n_layers: int
    ):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"Unknown model: {arch}")
        self.backbone = DNNBackbone(case.n_bus, case.n_gen, hidden_dim, n_layers)
        self.register_buffer("pg_max", torch.tensor(case.pg_max))
        self.register_buffer("r_max", torch.tensor(r_max))
        self.repair = arch != "dnn"
        self.dc_projection = DCProjection(case) if arch == "e2elrdc" else None

    def forward(
        self, pd: torch.Tensor, R: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pd : (batch, n_bus) float64 demand
            R  : (batch, 1)     float64 reserve requirement
        Returns:
            pg (batch, n_gen) and theta (batch, n_bus), both float64
        """
        z, theta = self.backbone(pd.float())
        pg = z.double() * self.pg_max
        theta = theta.double()
        if self.repair:
            pg = power_balance_repair(pg, self.pg_max, pd.sum(dim=-1, keepdim=True))
            pg = reserve_repair(pg, self.pg_max, self.r_max, R)
        if self.dc_projection is not None:
            theta = self.dc_projection(pg, pd, theta)
        return pg, theta
