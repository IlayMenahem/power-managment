"""
Training and evaluation of the dispatch proxies.

Implements:
  - Supervised Learning (SL) loss        (Eq. 10)
  - Self-Supervised Learning (SSL) loss  (Eq. 12)
    both extended with thermal and DC power flow penalties
  - Training loop with LR scheduling and early stopping
  - Evaluation metrics (optimality gap, feasibility rate, violations)

All physical quantities are float64 tensors (see models.py for why).

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2), Sections IV-V.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from data_utils import M_DC, M_PB, M_RES, M_TH, Case, Instances
from models import bus_injection
from torch.utils.data import DataLoader, TensorDataset

# Default constraint penalty weight per training mode (paper, Appendix B).
DEFAULT_LAM = {"ssl": 1.0, "sl": 1e-4}
# Validation always uses unit penalty weight, so that for SSL the validation
# loss equals the penalized objective used for the optimality gap.
VAL_LAM = 1.0


def resolve_device(name: str) -> str:
    """'auto' picks CUDA if available, else CPU. MPS is rejected: it lacks float64."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("mps"):
        raise ValueError("MPS does not support float64; use --device cpu or cuda.")
    return name


@dataclass(frozen=True)
class GridTensors:
    """Case constants on the training device (float64, indices as long)."""

    cost: torch.Tensor  # (n_gen,)
    pg_max: torch.Tensor  # (n_gen,)
    r_max: torch.Tensor  # (n_gen,)
    gen_bus: torch.Tensor  # (n_gen,)
    b_branch: torch.Tensor  # (n_branch,)
    rate: torch.Tensor  # (n_branch,)
    branch_from: torch.Tensor  # (n_branch,)
    branch_to: torch.Tensor  # (n_branch,)

    @staticmethod
    def from_case(case: Case, r_max: np.ndarray, device: str) -> "GridTensors":
        def t(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, device=device)

        return GridTensors(
            cost=t(case.cost),
            pg_max=t(case.pg_max),
            r_max=t(r_max),
            gen_bus=t(case.gen_bus),
            b_branch=t(case.b_branch),
            rate=t(case.rate),
            branch_from=t(case.branch_from),
            branch_to=t(case.branch_to),
        )


# ---------------------------------------------------------------------------
# Constraint violations and losses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violations:
    """Per-instance constraint violations, each of shape (batch,)."""

    power_balance: torch.Tensor  # |sum(pg) - D|
    reserve: torch.Tensor  # max(0, R - sum(min(r_max, pg_max - pg)))
    thermal: torch.Tensor  # sum over branches of max(0, |f| - rate)
    dc: torch.Tensor  # ||B theta - (pg_bus - pd)||_1

    def penalty(self) -> torch.Tensor:
        """Constraint penalty psi (Eq. 8), extended with thermal and DC terms."""
        return (
            M_PB * self.power_balance
            + M_RES * self.reserve
            + M_TH * self.thermal
            + M_DC * self.dc
        )


def compute_violations(
    pg: torch.Tensor,
    theta: torch.Tensor,
    pd: torch.Tensor,
    R: torch.Tensor,
    grid: GridTensors,
) -> Violations:
    flow = grid.b_branch * (theta[:, grid.branch_from] - theta[:, grid.branch_to])
    # B @ theta = A^T flow: each branch flow leaves its from-bus and enters its to-bus
    flow_out = (
        torch.zeros_like(theta)
        .index_add(1, grid.branch_from, flow)
        .index_add(1, grid.branch_to, -flow)
    )
    r_star = torch.min(grid.r_max, grid.pg_max - pg)
    return Violations(
        power_balance=(pg.sum(dim=-1) - pd.sum(dim=-1)).abs(),
        reserve=(R.squeeze(-1) - r_star.sum(dim=-1)).clamp(min=0.0),
        thermal=(flow.abs() - grid.rate).clamp(min=0.0).sum(dim=-1),
        dc=(flow_out - bus_injection(pg, pd, grid.gen_bus)).abs().sum(dim=-1),
    )


def proxy_loss(
    pg: torch.Tensor,
    theta: torch.Tensor,
    pd: torch.Tensor,
    R: torch.Tensor,
    pg_star: torch.Tensor,
    grid: GridTensors,
    mode: str,
    lam: float,
) -> torch.Tensor:
    """
    SSL: L = cost(pg) + lam * psi             (Eq. 12)
    SL : L = MAE(pg, pg_star) + lam * psi     (Eq. 10); instances without a
         ground-truth solution (NaN pg_star) are left out of the MAE.
    """
    penalty = compute_violations(pg, theta, pd, R, grid).penalty().mean()
    if mode == "ssl":
        return (pg @ grid.cost).mean() + lam * penalty
    valid = ~pg_star.isnan().any(dim=-1)
    mae = (pg[valid] - pg_star[valid]).abs().mean() if valid.any() else 0.0
    return mae + lam * penalty


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def build_datasets(
    instances: Instances,
    ground_truth: tuple[np.ndarray, np.ndarray] | None,
    device: str = "cpu",
) -> dict[str, TensorDataset]:
    """
    Train/val/test split (80/10/10, in order) of (pd, R, pg_star, obj_star),
    stored as float32. Without ground truth, pg_star and obj_star are NaN.
    """
    n = len(instances.pd)
    if ground_truth is None:
        n_gen = len(instances.r_max)
        ground_truth = (np.full((n, n_gen), np.nan), np.full(n, np.nan))
    pg_star, obj_star = ground_truth
    columns = [
        torch.tensor(x, dtype=torch.float32, device=device)
        for x in (instances.pd, instances.R[:, None], pg_star, obj_star)
    ]
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    bounds = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    return {
        name: TensorDataset(*(c[start:end] for c in columns))
        for name, (start, end) in bounds.items()
    }


def _unpack(
    batch: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pd, R, pg_star, obj_star) as float64."""
    pd, R, pg_star, obj_star = (t.double() for t in batch)
    return pd, R, pg_star, obj_star


def _batch_loss(
    model: torch.nn.Module,
    batch: list[torch.Tensor],
    grid: GridTensors,
    mode: str,
    lam: float,
) -> torch.Tensor:
    pd, R, pg_star, _ = _unpack(batch)
    pg, theta = model(pd, R)
    return proxy_loss(pg, theta, pd, R, pg_star, grid, mode, lam)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_model(
    model: torch.nn.Module,
    datasets: dict[str, TensorDataset],
    grid: GridTensors,
    mode: str = "ssl",
    lam: float = 1.0,
    lr: float = 1e-2,
    weight_decay: float = 1e-6,
    batch_size_train: int = 64,
    batch_size_eval: int = 256,
    max_epochs: int = 500,
    patience: int = 20,
    lr_patience: int = 10,
    max_time_min: float = 150.0,
    verbose: bool = True,
    epoch_callback: Callable[[int, float, float, float], None] | None = None,
) -> dict:
    """
    Train with SL or SSL, early-stopping on the validation loss. The model is
    left with its best validation weights. epoch_callback receives
    (epoch, train_loss, val_loss, elapsed_min) after every epoch.

    Returns the training history.
    """
    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size_train, shuffle=True
    )
    val_loader = DataLoader(datasets["val"], batch_size=batch_size_eval)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=lr_patience
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    def elapsed_min() -> float:
        return (time.time() - start_time) / 60.0

    for epoch in range(1, max_epochs + 1):
        if elapsed_min() > max_time_min:
            if verbose:
                print(f"  Time limit ({max_time_min} min) reached at epoch {epoch}.")
            break

        model.train()
        train_losses = []
        for batch in train_loader:
            loss = _batch_loss(model, batch, grid, mode, lam)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.detach())
        avg_train = torch.stack(train_losses).mean().item()

        model.eval()
        with torch.no_grad():
            val_losses = [
                _batch_loss(model, b, grid, mode, VAL_LAM) for b in val_loader
            ]
        avg_val = torch.stack(val_losses).mean().item()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 5 == 0:
            print(
                f"  Epoch {epoch:3d} | train {avg_train:.6f} | val {avg_val:.6f} | "
                f"lr {optimizer.param_groups[0]['lr']:.1e} | no_improve {epochs_no_improve}"
            )
        if epoch_callback is not None:
            epoch_callback(epoch, avg_train, avg_val, elapsed_min())

        if epochs_no_improve >= patience:
            if verbose:
                print(
                    f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)."
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(
            f"  Training done in {elapsed_min():.1f} min. Best val loss: {best_val_loss:.6f}"
        )
    history["best_val_loss"] = best_val_loss
    history["training_time_min"] = elapsed_min()
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: TensorDataset,
    grid: GridTensors,
    mode: str = "ssl",
    lam: float = 1.0,
    batch_size: int = 256,
    tol: float = 1e-4,
) -> dict[str, float]:
    """
    Metrics on a dataset:
      - test loss (training loss with penalty weight lam)
      - mean and shifted-geometric-mean optimality gap of the penalized
        objective, over instances with a ground-truth solution
      - feasibility rate: bounds, power balance, reserves and DC power flow
        all within tol (thermal limits are soft)
      - mean power balance violation, reserve shortage and DC violation
    """
    model.eval()
    losses, gaps, pb, reserve, dc, feasible = [], [], [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size):
        pd, R, pg_star, obj_star = _unpack(batch)
        pg, theta = model(pd, R)
        v = compute_violations(pg, theta, pd, R, grid)

        losses.append(proxy_loss(pg, theta, pd, R, pg_star, grid, mode, lam))
        z_hat = pg @ grid.cost + v.penalty()
        gaps.append((z_hat - obj_star) / obj_star.abs().clamp(min=1e-8))  # NaN w/o GT

        in_bounds = ((pg >= -tol) & (pg <= grid.pg_max + tol)).all(dim=-1)
        feasible.append(
            in_bounds & (v.power_balance < tol) & (v.reserve < tol) & (v.dc < tol)
        )
        pb.append(v.power_balance)
        reserve.append(v.reserve)
        dc.append(v.dc)

    def cat(xs: list[torch.Tensor]) -> np.ndarray:
        return torch.cat(xs).double().cpu().numpy()

    all_gaps = cat(gaps)
    valid_gaps = all_gaps[~np.isnan(all_gaps)]
    shift = 0.01  # shifted geometric mean with a 1% shift
    has_gaps = len(valid_gaps) > 0
    return {
        "test_loss": torch.stack(losses).mean().item(),
        "mean_gap_pct": 100.0 * valid_gaps.mean() if has_gaps else float("nan"),
        "sgm_gap_pct": 100.0 * (np.exp(np.mean(np.log(valid_gaps + shift))) - shift)
        if has_gaps
        else float("nan"),
        "feasibility_rate_pct": 100.0 * cat(feasible).mean(),
        "mean_pb_violation_pu": cat(pb).mean(),
        "mean_reserve_shortage_pu": cat(reserve).mean(),
        "mean_dc_violation_pu": cat(dc).mean(),
    }
