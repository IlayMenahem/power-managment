"""
Training and evaluation of the dispatch proxies (MLX version).

Implements:
  - Supervised Learning (SL) loss        (Eq. 10)
  - Self-Supervised Learning (SSL) loss  (Eq. 12)
  - Training loop with LR scheduling and early stopping
  - Evaluation metrics (optimality gap, feasibility rates, violations)

All physical quantities are float64 arrays, which MLX only supports on the CPU
stream, so every float64 computation runs inside `with mx.stream(mx.cpu)`
(see models.py for why float64 is needed).

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2), Sections IV-V.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from data_utils import M_PB, M_RES, M_TH, Case, Instances
from mlx.utils import tree_map
from models import ProxyOutput, bus_injection, f64

# Default constraint penalty weight per training mode (paper, Appendix B).
DEFAULT_LAM = {"ssl": 1.0, "sl": 1e-4}


def default_lam(mode: str, arch: str) -> float:
    """DEFAULT_LAM, except 0 for feasible models in SSL, where psi is zero anyway."""
    return 0.0 if mode == "ssl" and arch != "dnn" else DEFAULT_LAM[mode]

# Validation always uses unit penalty weight, so that for SSL the validation
# loss equals the penalized objective used for the optimality gap.
VAL_LAM = 1.0

# delta: e2elr_line trains its anchor to keep |flow| <= (1 - delta) rate.
ANCHOR_MARGIN = 0.05


@dataclass(frozen=True)
class GridTensors:
    """Case constants (float64, indices as int32)."""

    cost: mx.array  # (n_gen,)
    pg_min: mx.array  # (n_gen,)
    pg_max: mx.array  # (n_gen,)
    r_max: mx.array  # (n_gen,)
    gen_bus: mx.array  # (n_gen,)
    b_branch: mx.array  # (n_branch,)
    rate: mx.array  # (n_branch,)
    branch_from: mx.array  # (n_branch,)
    branch_to: mx.array  # (n_branch,)
    anchor_rate: mx.array  # (n_branch,) (1 - anchor_margin) * rate

    @staticmethod
    def from_case(
        case: Case, r_max: np.ndarray, anchor_margin: float = ANCHOR_MARGIN
    ) -> "GridTensors":
        def idx(x: np.ndarray) -> mx.array:
            return mx.array(x.astype(np.int32))

        return GridTensors(
            cost=f64(case.cost),
            pg_min=f64(case.pg_min),
            pg_max=f64(case.pg_max),
            r_max=f64(r_max),
            gen_bus=idx(case.gen_bus),
            b_branch=f64(case.b_branch),
            rate=f64(case.rate),
            branch_from=idx(case.branch_from),
            branch_to=idx(case.branch_to),
            anchor_rate=f64((1.0 - anchor_margin) * case.rate),
        )


# ---------------------------------------------------------------------------
# Constraint violations and losses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violations:
    """Per-instance constraint violations, each of shape (batch,)."""

    power_balance: mx.array  # |sum(pg) - D|
    reserve: mx.array  # max(0, R - sum(min(r_max, pg_max - pg)))
    thermal: mx.array  # sum over branches of max(0, |f| - rate)
    max_overload: mx.array  # max over branches of max(0, |f| - rate)
    dc: mx.array  # max over buses of |B theta - (pg_bus - pd)|, a sanity check
    # e2elr_line only (0 otherwise): the repaired anchor's overloads
    anchor_overload: mx.array  # sum over branches of max(0, |f_a| - anchor_rate)
    anchor_max_overload: mx.array  # max over branches of max(0, |f_a| - rate)

    def hard_penalty(self) -> mx.array:
        """Hard constraint penalty psi (Eq. 8)."""
        return M_PB * self.power_balance + M_RES * self.reserve

    def penalty(self) -> mx.array:
        """psi plus the thermal violation cost: the penalty of the optimality gap."""
        return self.hard_penalty() + M_TH * self.thermal


def compute_violations(
    out: ProxyOutput, pd: mx.array, R: mx.array, grid: GridTensors
) -> Violations:
    pg, theta = out.pg, out.theta
    flow = grid.b_branch * (theta[:, grid.branch_from] - theta[:, grid.branch_to])
    # B @ theta = A^T flow: each branch flow leaves its from-bus and enters its to-bus
    flow_out = (
        mx.zeros_like(theta)
        .at[:, grid.branch_from]
        .add(flow)
        .at[:, grid.branch_to]
        .add(-flow)
    )
    r_star = mx.minimum(grid.r_max, grid.pg_max - pg)
    overload = mx.maximum(mx.abs(flow) - grid.rate, 0.0)
    if out.anchor_flow is None:
        anchor_overload = anchor_max_overload = mx.zeros_like(overload[:, 0])
    else:
        abs_flow_a = mx.abs(out.anchor_flow)
        anchor_overload = mx.maximum(abs_flow_a - grid.anchor_rate, 0.0).sum(axis=-1)
        anchor_max_overload = mx.maximum(abs_flow_a - grid.rate, 0.0).max(axis=-1)
    return Violations(
        power_balance=mx.abs(pg.sum(axis=-1) - pd.sum(axis=-1)),
        reserve=mx.maximum(R.squeeze(-1) - r_star.sum(axis=-1), 0.0),
        thermal=overload.sum(axis=-1),
        max_overload=overload.max(axis=-1),
        dc=mx.abs(flow_out - bus_injection(pg, pd, grid.gen_bus)).max(axis=-1),
        anchor_overload=anchor_overload,
        anchor_max_overload=anchor_max_overload,
    )


def proxy_loss(
    out: ProxyOutput,
    pd: mx.array,
    R: mx.array,
    pg_star: mx.array,
    grid: GridTensors,
    mode: str,
    lam: float,
) -> mx.array:
    """
    SSL: L = cost(pg) + M_TH * (thermal + anchor) + lam * psi        (Eq. 12)
    SL : L = MAE(pg, pg_star) + lam * (psi + M_TH * (thermal + anchor))
         (Eq. 10, thermal weight mu = lam); instances without a ground-truth
         solution (NaN pg_star) are left out of the MAE.
    anchor is the anchor's overload of the tightened limits (1 - delta) rate
    (e2elr_line; 0 otherwise).
    """
    v = compute_violations(out, pd, R, grid)
    line_penalty = M_TH * (v.thermal + v.anchor_overload)
    pg = out.pg
    if mode == "ssl":
        objective = pg @ grid.cost + line_penalty
        return objective.mean() + lam * v.hard_penalty().mean()
    valid = ~mx.isnan(pg_star).any(axis=-1, keepdims=True)  # (batch, 1)
    err = mx.where(valid, mx.abs(pg - pg_star), 0.0)
    n_valid = valid.sum() * pg.shape[-1]
    penalty = v.hard_penalty() + line_penalty
    return err.sum() / mx.maximum(n_valid, 1) + lam * penalty.mean()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class Dataset:
    """(pd, R, pg_star, obj_star) as float32 numpy arrays."""

    def __init__(self, *columns: np.ndarray):
        self.columns = columns

    def __len__(self) -> int:
        return len(self.columns[0])

    def batches(
        self, batch_size: int, shuffle: bool = False
    ) -> Iterator[tuple[mx.array, ...]]:
        order = np.random.permutation(len(self)) if shuffle else np.arange(len(self))
        for i in range(0, len(self), batch_size):
            idx = order[i : i + batch_size]
            yield tuple(mx.array(c[idx]) for c in self.columns)


def build_datasets(
    instances: Instances, ground_truth: tuple[np.ndarray, np.ndarray] | None
) -> dict[str, Dataset]:
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
        np.asarray(x, dtype=np.float32)
        for x in (instances.pd, instances.R[:, None], pg_star, obj_star)
    ]
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    bounds = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    return {
        name: Dataset(*(c[start:end] for c in columns))
        for name, (start, end) in bounds.items()
    }


def _unpack(batch: tuple[mx.array, ...]) -> tuple[mx.array, ...]:
    """(pd, R, pg_star, obj_star) as float64. Call inside the CPU stream."""
    return tuple(t.astype(mx.float64) for t in batch)


def _batch_loss(
    model: nn.Module, batch: tuple[mx.array, ...], grid: GridTensors, mode: str, lam: float
) -> mx.array:
    with mx.stream(mx.cpu):
        pd, R, pg_star, _ = _unpack(batch)
        return proxy_loss(model(pd, R), pd, R, pg_star, grid, mode, lam)


class PlateauScheduler:
    """ReduceLROnPlateau(mode='min', factor, patience) with torch's 1e-4 rel threshold."""

    def __init__(self, optimizer: optim.Optimizer, factor: float, patience: int):
        self.optimizer, self.factor, self.patience = optimizer, factor, patience
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, metric: float) -> None:
        if metric < self.best * (1 - 1e-4):
            self.best, self.bad_epochs = metric, 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            self.optimizer.learning_rate = float(self.optimizer.learning_rate) * self.factor
            self.bad_epochs = 0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    datasets: dict[str, Dataset],
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
    optimizer = optim.Adam(learning_rate=lr)
    scheduler = PlateauScheduler(optimizer, factor=0.1, patience=lr_patience)

    def loss_fn(model, batch, lam):
        return _batch_loss(model, batch, grid, mode, lam)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(batch: tuple[mx.array, ...]) -> mx.array:
        loss, grads = loss_and_grad(model, batch, lam)
        # L2 weight decay added to the gradient, as torch.optim.Adam(weight_decay=...)
        grads = tree_map(lambda g, p: g + weight_decay * p, grads, model.trainable_parameters())
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        return loss

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
        train_losses = [
            train_step(b).item()
            for b in datasets["train"].batches(batch_size_train, shuffle=True)
        ]
        avg_train = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        for b in datasets["val"].batches(batch_size_eval):
            loss = _batch_loss(model, b, grid, mode, VAL_LAM)
            val_losses.append(loss.item())
        avg_val = float(np.mean(val_losses))

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.parameters()  # arrays are immutable: no copy needed
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 5 == 0:
            print(
                f"  Epoch {epoch:3d} | train {avg_train:.6f} | val {avg_val:.6f} | "
                f"lr {float(optimizer.learning_rate):.1e} | no_improve {epochs_no_improve}"
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
        model.update(best_state)
        mx.eval(model.parameters())

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


def shifted_geometric_mean(x: np.ndarray, shift: float) -> float:
    """exp(mean(log(x + s))) - s (Section VI-C); NaN for an empty array."""
    if len(x) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(x + shift))) - shift)


def violation_stats(violation: np.ndarray, tol: float) -> tuple[float, float]:
    """
    Table III statistics of one constraint: the percentage of predictions with
    violation <= tol, and the shifted geometric mean (shift 1 p.u.) of the
    violations of the other, infeasible, predictions (NaN if there are none).
    """
    violated = violation[violation > tol]
    return (
        100.0 * (1.0 - len(violated) / len(violation)),
        shifted_geometric_mean(violated, 1.0),
    )


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    grid: GridTensors,
    mode: str = "ssl",
    lam: float = 1.0,
    batch_size: int = 256,
    tol: float = 1e-4,
) -> dict[str, float]:
    """
    Metrics on a dataset:
      - test loss (training loss with penalty weight lam)
      - mean and shifted-geometric-mean (1% shift) optimality gap of the
        penalized objective, over instances with a ground-truth solution
      - feasibility rate: bounds, power balance, reserves and DC power flow
        (per bus) all within tol; thermal limits (soft) are not included
      - per hard constraint (power balance, reserves), the Table III
        statistics: % feasible and the shifted geometric mean of the
        violations of infeasible predictions (see violation_stats)
      - line feasibility rate (every |flow| <= rate + tol) and mean thermal
        violation
      - DC-OPF feasibility rate: feasible and every line within its hard limit
      - anchor line feasibility rate (e2elr_line; NaN otherwise)
      - mean largest per-bus DC residual (a sanity check, ~0)
    """
    model.eval()
    losses, gaps, pb, reserve, thermal, dc, feasible, line_ok, anchor_ok = (
        [] for _ in range(9)
    )
    has_anchor = False
    for batch in dataset.batches(batch_size):
        with mx.stream(mx.cpu):
            pd, R, pg_star, obj_star = _unpack(batch)
            out = model(pd, R)
            has_anchor = out.anchor_flow is not None
            v = compute_violations(out, pd, R, grid)

            loss = proxy_loss(out, pd, R, pg_star, grid, mode, lam)
            pg = out.pg
            z_hat = pg @ grid.cost + v.penalty()
            gap = (z_hat - obj_star) / mx.maximum(mx.abs(obj_star), 1e-8)  # NaN w/o GT
            in_bounds = (
                (pg >= grid.pg_min - tol) & (pg <= grid.pg_max + tol)
            ).all(axis=-1)
            ok = in_bounds & (v.power_balance < tol) & (v.reserve < tol) & (v.dc < tol)
            lines = v.max_overload < tol
            anchor_lines = v.anchor_max_overload < tol
            mx.eval(
                loss, gap, ok, lines, anchor_lines,
                v.power_balance, v.reserve, v.thermal, v.dc,
            )  # fmt: skip
        losses.append(loss.item())
        gaps.append(np.array(gap))
        feasible.append(np.array(ok))
        line_ok.append(np.array(lines))
        anchor_ok.append(np.array(anchor_lines))
        pb.append(np.array(v.power_balance))
        reserve.append(np.array(v.reserve))
        thermal.append(np.array(v.thermal))
        dc.append(np.array(v.dc))

    all_gaps = np.concatenate(gaps)
    valid_gaps = all_gaps[~np.isnan(all_gaps)]
    has_gaps = len(valid_gaps) > 0
    pb_feasible, pb_sgm = violation_stats(np.concatenate(pb), tol)
    res_feasible, res_sgm = violation_stats(np.concatenate(reserve), tol)
    all_lines_ok = np.concatenate(line_ok)
    return {
        "test_loss": float(np.mean(losses)),
        "mean_gap_pct": 100.0 * valid_gaps.mean() if has_gaps else float("nan"),
        "sgm_gap_pct": 100.0 * shifted_geometric_mean(valid_gaps, 0.01),
        "feasibility_rate_pct": 100.0 * np.concatenate(feasible).mean(),
        "pb_feasible_pct": pb_feasible,
        "pb_violation_sgm_pu": pb_sgm,
        "reserve_feasible_pct": res_feasible,
        "reserve_violation_sgm_pu": res_sgm,
        "line_feasibility_rate_pct": 100.0 * all_lines_ok.mean(),
        "opf_feasibility_rate_pct": 100.0
        * (np.concatenate(feasible) & all_lines_ok).mean(),
        "anchor_line_feasibility_rate_pct": (
            100.0 * np.concatenate(anchor_ok).mean() if has_anchor else float("nan")
        ),
        "mean_thermal_violation_pu": np.concatenate(thermal).mean(),
        "mean_max_dc_residual_pu": np.concatenate(dc).mean(),
    }
