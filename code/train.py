"""
Training and evaluation utilities for DNN and E2ELR models.

Implements:
  - Supervised Learning (SL) loss    (Eq. 10 in paper)
  - Self-Supervised Learning (SSL) loss  (Eq. 12 in paper)
  - Training loop with LR scheduling and early stopping
  - Evaluation metrics (optimality gap, feasibility rate, violations)

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2), Sections IV-V.
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Penalty / violation helpers
# ---------------------------------------------------------------------------

# MISO-based penalty prices in $/MW (Section V-C)
# Quantities in the loss (thermal flow, power balance, reserve) are in p.u.,
# so effective cost scales by baseMVA implicitly.
M_TH = 1500.0    # thermal violation penalty  (1500 $/MW)
M_PB = 3500.0    # power balance penalty      (3500 $/MW)
M_RES = 1100.0   # reserve shortage penalty   (1100 $/MW)


def compute_thermal_violations(pg, pd, ptdf_gen, ptdf_full, branch_rate):
    """
    Compute per-branch thermal violations: xi_e = max(0, |f_e| - f_max_e).

    Args:
        pg          : (batch, n_gen)
        pd          : (batch, n_bus)
        ptdf_gen    : (n_branch, n_gen) tensor
        ptdf_full   : (n_branch, n_bus) tensor
        branch_rate : (n_branch,) tensor

    Returns:
        xi : (batch, n_branch)  thermal violations
    """
    # Branch flow = ptdf_gen @ pg^T - ptdf_full @ pd^T  (each column is one instance)
    flow = (pg @ ptdf_gen.T) - (pd @ ptdf_full.T)  # (batch, n_branch)
    xi = torch.clamp(flow.abs() - branch_rate.unsqueeze(0), min=0.0)
    return xi


def compute_power_balance_violation(pg, D):
    """Absolute power balance violation: |sum(pg) - D|."""
    return (pg.sum(dim=-1, keepdim=True) - D).abs()


def compute_reserve_shortage(pg, pg_max, r_max, R):
    """Reserve shortage: max(0, R - sum min(r_max, pg_max - pg))."""
    r_star = torch.min(r_max.unsqueeze(0), pg_max.unsqueeze(0) - pg)
    total_reserve = r_star.sum(dim=-1, keepdim=True)
    return torch.clamp(R - total_reserve, min=0.0)


def compute_constraint_penalty(pg, D, pg_max, r_max, R, problem_type="ed"):
    """
    Constraint penalty psi(pg_hat) from Eq. 8 in the paper.
    For E2ELR this is zero (repair layers guarantee feasibility).
    """
    psi = M_PB * compute_power_balance_violation(pg, D)
    if problem_type == "edr":
        psi = psi + M_RES * compute_reserve_shortage(pg, pg_max, r_max, R)
    return psi  # (batch, 1)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def loss_sl(pg_hat, pg_star, pd, D, cost_coef, pg_max, r_max, R,
            ptdf_gen, ptdf_full, branch_rate,
            lam, mu, problem_type, is_feasible_model):
    """
    Supervised Learning loss (Eq. 10).

    L_SL = MAE(pg_hat, pg_star) + lambda * psi(pg_hat) + mu * M_th * ||xi||_1
    """
    n_gen = pg_hat.shape[1]

    # MAE on dispatch
    mae = (pg_hat - pg_star).abs().sum(dim=-1).mean() / n_gen

    # Thermal violations
    xi = compute_thermal_violations(pg_hat, pd, ptdf_gen, ptdf_full, branch_rate)
    thermal_pen = mu * M_TH * xi.sum(dim=-1).mean()

    # Constraint penalty (zero for feasible models)
    if is_feasible_model:
        constraint_pen = 0.0
    else:
        psi = compute_constraint_penalty(pg_hat, D, pg_max, r_max, R, problem_type)
        constraint_pen = lam * psi.mean()

    return mae + constraint_pen + thermal_pen


def loss_ssl(pg_hat, pd, D, cost_coef, pg_max, r_max, R,
             ptdf_gen, ptdf_full, branch_rate,
             lam, problem_type, is_feasible_model):
    """
    Self-Supervised Learning loss (Eq. 12).

    L_SSL = c(pg_hat) + M_th * ||xi||_1 + lambda * psi(pg_hat)
    """
    # Generation cost
    gen_cost = (cost_coef.unsqueeze(0) * pg_hat).sum(dim=-1).mean()

    # Thermal violations
    xi = compute_thermal_violations(pg_hat, pd, ptdf_gen, ptdf_full, branch_rate)
    thermal_pen = M_TH * xi.sum(dim=-1).mean()

    # Constraint penalty (zero for feasible models)
    if is_feasible_model:
        constraint_pen = 0.0
    else:
        psi = compute_constraint_penalty(pg_hat, D, pg_max, r_max, R, problem_type)
        constraint_pen = lam * psi.mean()

    return gen_cost + thermal_pen + constraint_pen


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------

def build_datasets(instances, pg_star, obj_star, case, problem_type, device="cpu"):
    """
    Build train/val/test TensorDatasets.

    Split: first 80% train, next 10% val, last 10% test.
    """
    pd_all = torch.tensor(instances["pd"], dtype=torch.float32, device=device)
    R_all = torch.tensor(instances["R_req"], dtype=torch.float32, device=device).unsqueeze(1)

    n = len(pd_all)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    if pg_star is not None:
        pg_all = torch.tensor(pg_star, dtype=torch.float32, device=device)
        obj_all = torch.tensor(obj_star, dtype=torch.float32, device=device)
    else:
        pg_all = torch.zeros(n, case["n_gen"], dtype=torch.float32, device=device)
        obj_all = torch.zeros(n, dtype=torch.float32, device=device)

    def _slice(t, s, e):
        return t[s:e]

    splits = {}
    for name, (s, e) in [("train", (0, n_train)),
                          ("val", (n_train, n_train + n_val)),
                          ("test", (n_train + n_val, n))]:
        splits[name] = TensorDataset(
            _slice(pd_all, s, e),
            _slice(R_all, s, e),
            _slice(pg_all, s, e),
            _slice(obj_all, s, e),
        )

    return splits


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_model(model, datasets, case, problem_type, mode="ssl",
                lam=0.0, mu=0.0,
                ptdf_gen_t=None, ptdf_full_t=None, branch_rate_t=None,
                cost_coef_t=None, pg_max_t=None, r_max_t=None,
                lr=1e-2, weight_decay=1e-6,
                batch_size_train=64, batch_size_eval=256,
                max_epochs=500, patience=20, lr_patience=10,
                max_time_min=150, device="cpu", verbose=True):
    """
    Train the model with SL or SSL.

    Args:
        model        : DNNModel or E2ELRModel
        datasets     : dict with 'train', 'val', 'test' TensorDatasets
        case         : case data dict
        problem_type : "ed" or "edr"
        mode         : "sl" or "ssl"
        lam          : constraint penalty weight
        mu           : thermal penalty weight (for SL; equals lam per paper)

    Returns:
        dict with training history (losses, best epoch, etc.)
    """
    is_feasible = hasattr(model, "power_balance") and model.power_balance is not None

    train_loader = DataLoader(datasets["train"], batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size_eval, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=lr_patience,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Check time limit
        elapsed_min = (time.time() - start_time) / 60.0
        if elapsed_min > max_time_min:
            if verbose:
                print(f"  Time limit ({max_time_min} min) reached at epoch {epoch}.")
            break

        # --- Training ---
        model.train()
        train_losses = []
        for batch in train_loader:
            pd_b, R_b, pg_star_b, obj_star_b = batch
            D_b = pd_b.sum(dim=-1, keepdim=True)

            # Forward pass
            if is_feasible:
                pg_hat = model(pd_b, D_b, R_b)
            else:
                pg_hat = model(pd_b)

            # Loss
            if mode == "sl":
                loss = loss_sl(
                    pg_hat, pg_star_b, pd_b, D_b,
                    cost_coef_t, pg_max_t, r_max_t, R_b,
                    ptdf_gen_t, ptdf_full_t, branch_rate_t,
                    lam, mu, problem_type, is_feasible,
                )
            else:
                loss = loss_ssl(
                    pg_hat, pd_b, D_b,
                    cost_coef_t, pg_max_t, r_max_t, R_b,
                    ptdf_gen_t, ptdf_full_t, branch_rate_t,
                    lam, problem_type, is_feasible,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pd_b, R_b, pg_star_b, obj_star_b = batch
                D_b = pd_b.sum(dim=-1, keepdim=True)

                if is_feasible:
                    pg_hat = model(pd_b, D_b, R_b)
                else:
                    pg_hat = model(pd_b)

                if mode == "sl":
                    loss = loss_sl(
                        pg_hat, pg_star_b, pd_b, D_b,
                        cost_coef_t, pg_max_t, r_max_t, R_b,
                        ptdf_gen_t, ptdf_full_t, branch_rate_t,
                        lam, mu, problem_type, is_feasible,
                    )
                else:
                    loss = loss_ssl(
                        pg_hat, pd_b, D_b,
                        cost_coef_t, pg_max_t, r_max_t, R_b,
                        ptdf_gen_t, ptdf_full_t, branch_rate_t,
                        lam, problem_type, is_feasible,
                    )
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        history["val_loss"].append(avg_val)

        scheduler.step(avg_val)

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train {avg_train:.6f} | val {avg_val:.6f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.1e} | no_improve {epochs_no_improve}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_min = (time.time() - start_time) / 60.0
    if verbose:
        print(f"  Training done in {elapsed_min:.1f} min. Best val loss: {best_val_loss:.6f}")

    history["best_val_loss"] = best_val_loss
    history["training_time_min"] = elapsed_min
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, dataset, case, problem_type,
                   ptdf_gen_t, ptdf_full_t, branch_rate_t,
                   cost_coef_t, pg_max_t, r_max_t,
                   batch_size=256, tol=1e-4, device="cpu"):
    """
    Evaluate model on a dataset and compute:
      - Mean optimality gap
      - Feasibility rate (% satisfying all hard constraints)
      - Mean power balance violation
      - Mean reserve shortage (for ED-R)

    Returns dict with metrics.
    """
    is_feasible = hasattr(model, "power_balance") and model.power_balance is not None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_gaps = []
    all_pb_viol = []
    all_res_short = []
    all_feasible = []

    for batch in loader:
        pd_b, R_b, pg_star_b, obj_star_b = batch
        D_b = pd_b.sum(dim=-1, keepdim=True)

        if is_feasible:
            pg_hat = model(pd_b, D_b, R_b)
        else:
            pg_hat = model(pd_b)

        # --- Compute penalized objective for predictions ---
        gen_cost = (cost_coef_t.unsqueeze(0) * pg_hat).sum(dim=-1)

        xi = compute_thermal_violations(pg_hat, pd_b, ptdf_gen_t, ptdf_full_t, branch_rate_t)
        thermal = M_TH * xi.sum(dim=-1)

        pb_viol = compute_power_balance_violation(pg_hat, D_b).squeeze(-1)
        pb_pen = M_PB * pb_viol

        if problem_type == "edr":
            res_short = compute_reserve_shortage(pg_hat, pg_max_t, r_max_t, R_b).squeeze(-1)
            res_pen = M_RES * res_short
        else:
            res_short = torch.zeros_like(pb_viol)
            res_pen = torch.zeros_like(pb_viol)

        z_hat = gen_cost + thermal + pb_pen + res_pen
        z_star = obj_star_b

        # Optimality gap (only for instances with valid ground truth)
        valid = ~torch.isnan(z_star)
        gaps = torch.where(valid, (z_hat - z_star) / z_star.abs().clamp(min=1e-8), torch.zeros_like(z_hat))

        # Feasibility check
        bounds_ok = (pg_hat >= -tol).all(dim=-1) & (pg_hat <= pg_max_t.unsqueeze(0) + tol).all(dim=-1)
        pb_ok = pb_viol < tol
        if problem_type == "edr":
            res_ok = res_short < tol
            feasible = bounds_ok & pb_ok & res_ok
        else:
            feasible = bounds_ok & pb_ok

        all_gaps.append(gaps)
        all_pb_viol.append(pb_viol)
        all_res_short.append(res_short)
        all_feasible.append(feasible)

    all_gaps = torch.cat(all_gaps).cpu().numpy()
    all_pb_viol = torch.cat(all_pb_viol).cpu().numpy()
    all_res_short = torch.cat(all_res_short).cpu().numpy()
    all_feasible = torch.cat(all_feasible).cpu().numpy()

    # Shifted geometric mean for gaps (shift s=0.01 i.e. 1%)
    s = 0.01
    valid_gaps = all_gaps[~np.isnan(all_gaps)]
    if len(valid_gaps) > 0:
        sgm_gap = np.exp(np.mean(np.log(valid_gaps + s))) - s
    else:
        sgm_gap = float("nan")

    results = {
        "mean_gap_pct": 100.0 * np.nanmean(valid_gaps) if len(valid_gaps) > 0 else float("nan"),
        "sgm_gap_pct": 100.0 * sgm_gap,
        "feasibility_rate_pct": 100.0 * np.mean(all_feasible),
        "mean_pb_violation_pu": np.mean(all_pb_viol),
        "mean_reserve_shortage_pu": np.mean(all_res_short),
    }

    return results
