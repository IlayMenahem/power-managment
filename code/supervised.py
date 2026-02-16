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
from data_generation import load_reference_case, make_grain_dataset
from model import E2ELR, e2elr_batched
from utils import save_checkpoint


def loss_and_accuracy(
    model: E2ELR,
    d_batch: jnp.ndarray,
    p_max_batch: jnp.ndarray,
    r_max_batch: jnp.ndarray,
    R_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute surrogate cost (loss) and accuracy for E2ELR on a batch.

    Loss = mean over batch of sum_g(p_hat_g^2) (differentiable surrogate cost).
    Accuracy = -loss so that higher is better for early stopping.

    Args:
        model: E2ELR model.
        d_batch: Demand (batch, n_buses).
        p_max_batch: Max generation (batch, n_gens).
        r_max_batch: Max reserve (batch, n_gens).
        R_batch: Reserve requirement (batch,) or scalar.

    Returns:
        loss: Scalar loss (mean surrogate cost).
        accuracy: -loss (for monitoring / early stopping).
    """
    p_hat = e2elr_batched(model, d_batch, p_max_batch, r_max_batch, R_batch)
    # Surrogate cost: sum of squares of dispatch (differentiable, encourages spread)
    per_sample_cost = jnp.sum(p_hat**2, axis=-1)
    loss = jnp.mean(per_sample_cost)
    accuracy = -loss  # higher is better for early stopping
    return loss, accuracy


def train_model(
    model: E2ELR,
    train_dataset,
    val_dataset,
    num_epochs: int,
    optimizer: optax.GradientTransformation,
    loss_and_accuracy_fn: Callable,
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
        loss_and_accuracy_fn: (model, d, p_max, r_max, R) -> (loss, accuracy).
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
    ) -> tuple[E2ELR, optax.OptState, jnp.ndarray, jnp.ndarray]:
        def loss_fn(mdl):
            return loss_and_accuracy_fn(mdl, d, p_max, r_max, R)

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
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return loss_and_accuracy_fn(m, d, p_max, r_max, R)

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
            m, opt_state, loss, acc = make_step(m, opt_state, d, p_max, r_max, R)
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
            loss, acc = eval_step(m, d, p_max, r_max, R)
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
    d_ref, p_max_ref = load_reference_case(ref_path)

    n_buses = d_ref.shape[0]
    n_gens = p_max_ref.shape[0]

    checkpoint_dir = os.path.join("models", "checkpoints")
    num_epochs = 100
    batch_size = 128
    num_train = 2**14
    num_val = 2**12
    early_stopping_patience = 5
    min_delta = 1e-3

    num_layers = 3
    hidden_size = 256
    dropout_rate = 0.2

    key = jax.random.PRNGKey(0)
    model = E2ELR(
        in_size=n_buses,
        out_size=n_gens,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
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
        checkpoint_dir,
        early_stopping_patience,
        min_delta,
    )

    print("Training finished.")
    print(f"Best val accuracy (negative loss): {float(max(accuracy_val)):.4f}")
