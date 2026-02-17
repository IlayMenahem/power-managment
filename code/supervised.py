"""
Supervised (self-supervised) training for E2ELR.

Trains the E2ELR model on (d, p_max, r_max, R) instances from data_generation
by minimizing a differentiable surrogate cost (e.g. sum of squares of dispatch).
No optimal solutions required; repair layers ensure feasibility.
"""

from __future__ import annotations

import os
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import vmap

from data_generation import load_reference_case, make_grain_dataset
from model import E2ELR, e2elr_batched, thermal_violations_l1
from utils import save_checkpoint


def _reserve_shortage(p: jnp.ndarray, p_max: jnp.ndarray, r_max: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Reserve shortage xi_r = max(0, R - sum_g min(r_max_g, p_max_g - p_g)). Paper Eq. (77)."""
    available = jnp.minimum(r_max, p_max - p)
    total_reserve = jnp.sum(available, axis=-1)
    return jnp.maximum(R - total_reserve, 0.0)


def loss_and_accuracy(
    model: E2ELR,
    d_batch: jnp.ndarray,
    p_max_batch: jnp.ndarray,
    r_max_batch: jnp.ndarray,
    R_batch: jnp.ndarray,
    ref: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute SSL loss (paper Eq. 105-106) and accuracy for E2ELR on a batch.

    Loss = mean over batch of phi^SSL(p_hat) + lambda * psi(p_hat), where
    phi^SSL(p_hat) = c(p_hat) + Mth * ||xi_th(p_hat)||_1,
    psi(p_hat) = M_pb * |e'@d - e'@p_hat| + M_res * xi_r(p_hat),
    xi_r(p_hat) = max(0, R - sum_g min(r_max_g, p_max_g - p_hat_g)).
    Accuracy = -loss so that higher is better for early stopping.

    Args:
        model: E2ELR model.
        d_batch: Demand (batch, n_buses).
        p_max_batch: Max generation (batch, n_gens).
        r_max_batch: Max reserve (batch, n_gens).
        R_batch: Reserve requirement (batch,) or scalar.
        ref: Dict with c_linear, Phi, PTDF_bus, f_min, f_max, Mth, and optionally
            M_pb, M_res (default 1500, 1100 from paper).

    Returns:
        loss: Scalar loss.
        accuracy: -loss (for monitoring / early stopping).
    """
    p_hat = e2elr_batched(model, d_batch, p_max_batch, r_max_batch, R_batch)

    c_linear = ref["c_linear"]
    Phi = ref["Phi"]
    PTDF_bus = ref["PTDF_bus"]
    f_min = ref["f_min"]
    f_max = ref["f_max"]
    Mth = ref["Mth"]
    M_pb = ref.get("M_pb", 1500.0)
    M_res = ref.get("M_res", 1100.0)
    lam = ref.get("lambda", 1.0)

    # phi^SSL: c(p_hat) + Mth * xi_th per sample
    per_sample_cost = jnp.sum(c_linear * p_hat, axis=-1)
    per_sample_xi_th = vmap(
        lambda p, d: thermal_violations_l1(p, d, Phi, PTDF_bus, f_min, f_max),
        in_axes=(0, 0),
        out_axes=0,
    )(p_hat, d_batch)
    phi_ssl = per_sample_cost + Mth * per_sample_xi_th

    # psi: power balance + reserve (paper Eq. 70-77)
    sum_d = jnp.sum(d_batch, axis=-1)
    sum_p = jnp.sum(p_hat, axis=-1)
    power_balance_viol = jnp.abs(sum_d - sum_p)
    
    R_flat = jnp.broadcast_to(R_batch, (d_batch.shape[0],)) if R_batch.ndim == 0 else R_batch
    xi_r = _reserve_shortage(p_hat, p_max_batch, r_max_batch, R_flat)
    psi = M_pb * power_balance_viol + M_res * xi_r

    per_sample_loss = phi_ssl + lam * psi
    loss = jnp.mean(per_sample_loss)

    accuracy = -loss  # higher is better for early stopping
    return loss, accuracy


def train_model(
    model: E2ELR,
    train_dataset,
    val_dataset,
    num_epochs: int,
    optimizer: optax.GradientTransformation,
    loss_and_accuracy_fn: Callable,
    ref: dict,
    checkpoint_dir: str | None = None,
    early_stopping_patience: int | None = None,
    min_delta: float = 0.0,
) -> tuple[E2ELR, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Train the E2ELR model.

    Args:
        model: E2ELR model.
        train_dataset: Iterable of (d, p_max, r_max, R) batches.
        val_dataset: Iterable of (d, p_max, r_max, R) batches for validation.
        num_epochs: Number of epochs.
        optimizer: Optax optimizer.
        loss_and_accuracy_fn: (model, d, p_max, r_max, R, ref) -> (loss, accuracy).
        ref: Dict (c_linear, Phi, PTDF_bus, f_min, f_max, Mth, optionally M_pb, M_res) for SSL loss.
        checkpoint_dir: Directory to save checkpoints; None = no checkpoints.
        early_stopping_patience: Stop if val accuracy does not improve for this many epochs.
        min_delta: Minimum improvement in val accuracy to reset early-stopping counter.

    Returns:
        model: Trained model.
        train_losses, train_accuracies, val_losses, val_accuracies: Arrays.
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))  # type: ignore[arg-type]

    @eqx.filter_jit
    def make_step(
        m: E2ELR,
        opt_state: optax.OptState,
        d: jnp.ndarray,
        p_max: jnp.ndarray,
        r_max: jnp.ndarray,
        R: jnp.ndarray,
        ref_dict: dict,
    ) -> tuple[E2ELR, optax.OptState, jnp.ndarray, jnp.ndarray]:
        def loss_fn(mdl):
            return loss_and_accuracy_fn(mdl, d, p_max, r_max, R, ref_dict)

        (loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(m)
        updates, opt_state = optimizer.update(grads, opt_state, m)  # type: ignore[arg-type]
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss, acc

    def eval_step(
        m: E2ELR,
        d: jnp.ndarray,
        p_max: jnp.ndarray,
        r_max: jnp.ndarray,
        R: jnp.ndarray,
        ref_dict: dict,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return loss_and_accuracy_fn(m, d, p_max, r_max, R, ref_dict)

    def train_epoch(m: E2ELR, opt_state: optax.OptState) -> tuple[E2ELR, optax.OptState, jnp.ndarray, jnp.ndarray]:
        epoch_losses = []
        epoch_accs = []
        for batch in train_dataset:
            d, p_max, r_max, R = batch
            d = jnp.asarray(d)
            p_max = jnp.asarray(p_max)
            r_max = jnp.asarray(r_max)
            R = jnp.asarray(R)
            if R.ndim == 0:
                R = jnp.broadcast_to(R, (d.shape[0],))
            m, opt_state, loss, acc = make_step(m, opt_state, d, p_max, r_max, R, ref)
            epoch_losses.append(loss)
            epoch_accs.append(acc)
        return m, opt_state, jnp.mean(jnp.array(epoch_losses)), jnp.mean(jnp.array(epoch_accs))

    def validate_epoch(m: E2ELR) -> tuple[jnp.ndarray, jnp.ndarray]:
        epoch_losses = []
        epoch_accs = []
        for batch in val_dataset:
            d, p_max, r_max, R = batch
            d = jnp.asarray(d)
            p_max = jnp.asarray(p_max)
            r_max = jnp.asarray(r_max)
            R = jnp.asarray(R)
            if R.ndim == 0:
                R = jnp.broadcast_to(R, (d.shape[0],))
            loss, acc = eval_step(m, d, p_max, r_max, R, ref)
            epoch_losses.append(loss)
            epoch_accs.append(acc)
        return jnp.mean(jnp.array(epoch_losses)), jnp.mean(jnp.array(epoch_accs))

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = float("-inf")
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model, opt_state, t_loss, t_acc = train_epoch(model, opt_state)
        v_loss, v_acc = validate_epoch(model)

        train_losses.append(t_loss)
        train_accuracies.append(t_acc)
        val_losses.append(v_loss)
        val_accuracies.append(v_acc)

        val_acc_value = float(v_acc)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {float(t_loss):.4f}, Train Acc: {float(t_acc):.4f}, "
            f"Val Loss: {float(v_loss):.4f}, Val Acc: {val_acc_value:.4f}"
        )

        improved = val_acc_value > best_val_acc + min_delta
        if improved:
            best_val_acc = val_acc_value
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if checkpoint_dir:
            save_checkpoint(
                model,
                opt_state,
                checkpoint_dir,
                "last",
                epoch + 1,
                {"val_loss": float(v_loss), "val_acc": val_acc_value},
            )
            if improved:
                save_checkpoint(
                    model,
                    opt_state,
                    checkpoint_dir,
                    "best",
                    epoch + 1,
                    {"val_loss": float(v_loss), "val_acc": val_acc_value},
                )

        if (
            early_stopping_patience is not None
            and no_improve_epochs >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1} (no val accuracy improvement > {min_delta} for {early_stopping_patience} epochs)."
            )
            break

    return (
        model,
        jnp.stack(train_losses),
        jnp.stack(train_accuracies),
        jnp.stack(val_losses),
        jnp.stack(val_accuracies),
    )


if __name__ == "__main__":
    ref_path = os.path.join("data", "pglib_opf_case300_ieee_ref.npz")
    d_ref, p_max_ref, ref = load_reference_case(ref_path)
    if ref is None:
        raise ValueError(
            "Reference case must include SSL ref dict (c_linear, Phi, PTDF_bus, f_min, f_max, Mth). "
            f"File {ref_path} is missing these keys."
        )

    n_buses = d_ref.shape[0]
    n_gens = p_max_ref.shape[0]

    checkpoint_dir = os.path.join("models", "checkpoints")
    num_epochs = 100
    batch_size = 128
    num_train = 40000
    num_val = 5000
    early_stopping_patience = 5
    min_delta = 1e-4

    num_layers = 5
    hidden_size = 1024

    key = jax.random.PRNGKey(0)
    model = E2ELR(
        in_size=n_buses,
        out_size=n_gens,
        num_layers=num_layers,
        hidden_size=hidden_size,
        key=key,
    )

    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)

    train_dataset = make_grain_dataset(
        d_ref,
        p_max_ref,
        num_train,
        mode="edr",
        seed=0,
        shuffle_seed=42,
        batch_size=batch_size,
    )
    val_dataset = make_grain_dataset(
        d_ref,
        p_max_ref,
        num_val,
        mode="edr",
        seed=1,
        shuffle_seed=43,
        batch_size=batch_size,
    )

    model, losses_train, accuracy_train, losses_val, accuracy_val = train_model(
        model,
        train_dataset,
        val_dataset,
        num_epochs,
        optimizer,
        loss_and_accuracy,
        ref,
        checkpoint_dir,
        early_stopping_patience,
        min_delta,
    )

    print("Training finished.")
    print(f"Best val accuracy (negative loss): {float(max(accuracy_val)):.4f}")
