"""
Neural network models for Economic Dispatch optimization proxies.

Implements:
  - DNN:   Vanilla fully-connected network (baseline, no feasibility layers)
  - E2ELR: End-to-End Learning and Repair (DNN + Power Balance + Reserve repair)

All models return (pg, theta) where theta = B_pinv @ (pg_bus - pd) is the
minimum-norm DC voltage angle satisfying the power flow equations.

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2), Sections III-IV.
"""

import torch
import torch.nn as nn

EPS = 1e-8


# ---------------------------------------------------------------------------
# Shared theta computation
# ---------------------------------------------------------------------------

def compute_theta(pg, pd, B_pinv_t, gen_bus_idx_t):
    """
    Compute minimum-norm DC voltage angles: theta = B_pinv @ (pg_bus - pd).

    Args:
        pg           : (batch, n_gen)  generator dispatch
        pd           : (batch, n_bus)  bus demand
        B_pinv_t     : (n_bus, n_bus)  pseudoinverse of B_bus (tensor)
        gen_bus_idx_t: (n_gen,) long   bus index for each generator

    Returns:
        theta : (batch, n_bus)
    """
    batch_size = pg.shape[0]
    n_bus = pd.shape[1]
    # Scatter pg onto buses
    pg_bus = torch.zeros(batch_size, n_bus, device=pg.device, dtype=pg.dtype)
    pg_bus.scatter_add_(1, gen_bus_idx_t.unsqueeze(0).expand(batch_size, -1), pg)
    net = pg_bus - pd                   # (batch, n_bus) net injection
    theta = net @ B_pinv_t.T            # (batch, n_bus)  B_pinv is symmetric, but .T is safe
    return theta


# ---------------------------------------------------------------------------
# Repair Layers (differentiable, closed-form)
# ---------------------------------------------------------------------------

class PowerBalanceRepair(nn.Module):
    """
    Power Balance Repair Layer (Eq. 6-7 in paper).

    Given pg in [0, pg_max], outputs pg_tilde that satisfies:
        sum(pg_tilde) = D   and   0 <= pg_tilde <= pg_max

    Uses proportional response: if total generation is too low, each generator
    increases by a fraction of its upward headroom; if too high, decreases by
    a fraction of its current output.
    """

    def forward(self, pg, pg_max, D):
        """
        Args:
            pg     : (batch, n_gen) initial dispatch in [0, pg_max]
            pg_max : (n_gen,) or (batch, n_gen) maximum generation
            D      : (batch, 1) total demand
        Returns:
            pg_repaired : (batch, n_gen)
        """
        total_gen = pg.sum(dim=-1, keepdim=True)        # (batch, 1)
        total_max = pg_max.sum(dim=-1, keepdim=True) if pg_max.dim() > 1 \
            else pg_max.sum().unsqueeze(0).unsqueeze(0)  # scalar -> (1, 1)

        # Shortage: need to increase generation
        eta_up = (D - total_gen) / (total_max - total_gen + EPS)
        eta_up = eta_up.clamp(0.0, 1.0)

        # Surplus: need to decrease generation
        eta_down = (total_gen - D) / (total_gen + EPS)
        eta_down = eta_down.clamp(0.0, 1.0)

        shortage = (total_gen < D)  # (batch, 1) bool

        pg_up = (1 - eta_up) * pg + eta_up * pg_max     # move toward pg_max
        pg_down = (1 - eta_down) * pg                    # move toward 0

        pg_repaired = torch.where(shortage, pg_up, pg_down)
        return pg_repaired


class ReserveRepair(nn.Module):
    """
    Reserve Repair Layer (Algorithm 1 in paper).

    Given pg in S_D (satisfies bounds + power balance), adjusts dispatch
    so that sufficient reserves can be allocated.  Power balance is maintained
    because increases and decreases are balanced.
    """

    def forward(self, pg, pg_max, r_max, R):
        """
        Args:
            pg     : (batch, n_gen) dispatch satisfying bounds + power balance
            pg_max : (n_gen,) or (batch, n_gen) max generation
            r_max  : (n_gen,) or (batch, n_gen) max reserve per generator
            R      : (batch, 1) reserve requirement
        Returns:
            pg_adj : (batch, n_gen) adjusted dispatch
        """
        # Maximum achievable reserve per generator
        r_star = torch.min(r_max, pg_max - pg)                       # (batch, n_gen)
        delta_R = R - r_star.sum(dim=-1, keepdim=True)               # reserve shortage

        # Target dispatch = pg_max - r_max (where full reserve is available)
        target = pg_max - r_max                                       # (n_gen,) or (batch, n_gen)

        # Split generators into two groups
        G_up = (pg <= target).float()    # can increase dispatch without losing reserves
        G_down = (pg > target).float()   # must decrease dispatch to free up reserves

        # Headroom for each group
        Delta_up = ((target - pg) * G_up).sum(dim=-1, keepdim=True)
        Delta_down = ((pg - target) * G_down).sum(dim=-1, keepdim=True)

        # How much to adjust (line 6 of Algorithm 1)
        Delta = torch.clamp(
            torch.min(torch.min(delta_R, Delta_up), Delta_down),
            min=0.0,
        )

        # Proportional weights (line 7)
        alpha_up = Delta / (Delta_up + EPS)
        alpha_down = Delta / (Delta_down + EPS)

        # Adjust dispatches (line 8): move each generator toward 'target'
        pg_adj_up = (1 - alpha_up) * pg + alpha_up * target
        pg_adj_down = (1 - alpha_down) * pg + alpha_down * target

        pg_adj = torch.where(pg <= target, pg_adj_up, pg_adj_down)
        return pg_adj


# ---------------------------------------------------------------------------
# DNN Backbone
# ---------------------------------------------------------------------------

class DNNBackbone(nn.Module):
    """
    Fully-connected DNN with ReLU, BatchNorm, and Dropout.

    Input  : pd (n_bus,)
    Output : z  (n_gen,) in [0, 1] via sigmoid
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=3, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, pd):
        return self.net(pd)


# ---------------------------------------------------------------------------
# Model Wrappers
# ---------------------------------------------------------------------------

class DNNModel(nn.Module):
    """
    Vanilla DNN baseline (Figure 2a in paper).

    Predicts pg = sigmoid(DNN(pd)) * pg_max, then computes theta = B_pinv @ (pg_bus - pd).
    No feasibility layers — only enforces generation bounds [0, pg_max].

    Returns (pg, theta).
    """

    def __init__(self, n_bus, n_gen, pg_max, hidden_dim=256, n_layers=3,
                 B_pinv=None, gen_bus_idx=None):
        super().__init__()
        self.backbone = DNNBackbone(n_bus, n_gen, hidden_dim, n_layers)
        self.register_buffer("pg_max", torch.tensor(pg_max, dtype=torch.float32))
        self.register_buffer("B_pinv", torch.tensor(B_pinv, dtype=torch.float32))
        self.register_buffer("gen_bus_idx", torch.tensor(gen_bus_idx, dtype=torch.long))

    def forward(self, pd):
        z = self.backbone(pd)             # (batch, n_gen) in [0, 1]
        pg = z * self.pg_max              # (batch, n_gen) in [0, pg_max]
        theta = compute_theta(pg, pd, self.B_pinv, self.gen_bus_idx)
        return pg, theta


class E2ELRModel(nn.Module):
    """
    End-to-End Learning and Repair model (Figure 2e / Figure 3 in paper).

    Pipeline: sigmoid(DNN(pd)) * pg_max -> PowerBalanceRepair -> ReserveRepair -> theta.
    Output satisfies all hard constraints (bounds, power balance, reserves).
    Theta is exact (DC residual = 0) when power balance holds.

    Returns (pg, theta).
    """

    def __init__(self, n_bus, n_gen, pg_max, hidden_dim=256, n_layers=3,
                 problem_type="ed", r_max=None, B_pinv=None, gen_bus_idx=None):
        super().__init__()
        self.backbone = DNNBackbone(n_bus, n_gen, hidden_dim, n_layers)
        self.register_buffer("pg_max", torch.tensor(pg_max, dtype=torch.float32))
        self.power_balance = PowerBalanceRepair()
        self.problem_type = problem_type
        self.register_buffer("B_pinv", torch.tensor(B_pinv, dtype=torch.float32))
        self.register_buffer("gen_bus_idx", torch.tensor(gen_bus_idx, dtype=torch.long))

        if problem_type == "edr" and r_max is not None:
            self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float32))
            self.reserve_repair = ReserveRepair()
        else:
            self.r_max = None
            self.reserve_repair = None

    def forward(self, pd, D, R=None):
        """
        Args:
            pd : (batch, n_bus)  demand vector
            D  : (batch, 1)      total demand = pd.sum(dim=-1, keepdim=True)
            R  : (batch, 1)      reserve requirement (only for ED-R)
        Returns:
            (pg, theta) : ((batch, n_gen), (batch, n_bus))
        """
        z = self.backbone(pd)                          # (batch, n_gen) in [0, 1]
        pg = z * self.pg_max                           # enforce bounds

        pg = self.power_balance(pg, self.pg_max, D)    # enforce power balance

        if self.problem_type == "edr" and R is not None and self.reserve_repair is not None:
            pg = self.reserve_repair(pg, self.pg_max, self.r_max, R)

        theta = compute_theta(pg, pd, self.B_pinv, self.gen_bus_idx)
        return pg, theta
