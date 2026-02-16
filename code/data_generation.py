"""
Data generation for E2ELR (paper Sec. 5.1).

Generates ED and ED-R instances by perturbing a reference load profile.
Training is self-supervised only; no labels (optimal solutions) are generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


# Log-normal with mean 1 and std 0.05: sigma^2 = ln(1 + 0.05^2), mu = -sigma^2/2
_LOG_NORMAL_SIGMA_SQ = float(np.log(1.0 + 0.05**2))
_LOG_NORMAL_MU = -0.5 * _LOG_NORMAL_SIGMA_SQ


def _sample_demand(key: jax.Array, d_ref: jnp.ndarray) -> jnp.ndarray:
    """Sample perturbed demand d = gamma * eta * d_ref (element-wise)."""
    key_gamma, key_eta = jr.split(key)
    # gamma ~ U[0.8, 1.2]
    gamma = jr.uniform(key_gamma, minval=0.8, maxval=1.2)
    # eta per bus: log-normal(mean=1, std=0.05) -> Z ~ N(mu, sigma^2), eta = exp(Z)
    n_buses = d_ref.shape[0]
    z = jr.normal(key_eta, shape=(n_buses,)) * np.sqrt(_LOG_NORMAL_SIGMA_SQ) + _LOG_NORMAL_MU
    eta = jnp.exp(z)
    return (gamma * eta * d_ref).astype(d_ref.dtype)


def sample_ed_instance(
    key: jax.Array,
    d_ref: jnp.ndarray,
    p_max: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """
    Sample a single ED instance (no reserve).

    Returns:
        (d, p_max, r_max, R) with r_max = 0, R = 0.
    """
    d = _sample_demand(key, d_ref)
    r_max = jnp.zeros_like(p_max)
    R = 0.0

    return d, p_max, r_max, R


def sample_edr_instance(
    key: jax.Array,
    d_ref: jnp.ndarray,
    p_max: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """
    Sample a single ED-R instance (with reserve).

    alpha_r = 5 * max(p_max) / sum(p_max), r_max = alpha_r * p_max,
    R ~ U(1, 2) * max(p_max).
    """
    key_d, key_R = jr.split(key)
    d = _sample_demand(key_d, d_ref)
    p_max_sum = jnp.sum(p_max)
    p_max_inf = jnp.max(p_max)
    alpha_r = 5.0 * p_max_inf / jnp.maximum(p_max_sum, 1e-12)
    r_max = (alpha_r * p_max).astype(p_max.dtype)
    # R ~ U(1, 2) * max(p_max)
    R = float(jr.uniform(key_R, minval=1.0, maxval=2.0) * p_max_inf)
    return d, p_max, r_max, R


def generate_dataset(
    key: jax.Array,
    d_ref: jnp.ndarray,
    p_max: jnp.ndarray,
    n_total: int = 50_000,
    train_size: int = 40_000,
    val_size: int = 5_000,
    test_size: int = 5_000,
    mode: Literal["ed", "edr"] = "edr",
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Generate train/val/test datasets (delegates to vectorized implementation).

    Returns dict with keys "train", "val", "test". Each value is
    (d_batch, p_max_batch, r_max_batch, R_batch) where p_max/r_max are
    broadcast per instance (same grid).
    """
    return generate_dataset_vectorized(
        key, d_ref, p_max, n_total, train_size, val_size, test_size, mode
    )


def generate_dataset_vectorized(
    key: jax.Array,
    d_ref: jnp.ndarray,
    p_max: jnp.ndarray,
    n_total: int = 50_000,
    train_size: int = 40_000,
    val_size: int = 5_000,
    test_size: int = 5_000,
    mode: Literal["ed", "edr"] = "edr",
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Generate train/val/test datasets (vectorized over instances).

    Same contract as generate_dataset but uses vmap for efficiency.
    """
    if train_size + val_size + test_size != n_total:
        raise ValueError(
            f"train_size + val_size + test_size must equal n_total: "
            f"{train_size} + {val_size} + {test_size} != {n_total}"
        )
    n_gen = p_max.shape[0]

    key_train, key_val, key_test = jr.split(key, 3)

    def sample_d_batch(key: jax.Array, batch_size: int) -> jnp.ndarray:
        key_g, key_eta = jr.split(key)
        gammas = jr.uniform(
            key_g, shape=(batch_size,), minval=0.8, maxval=1.2
        )
        n_buses = d_ref.shape[0]
        z = (
            jr.normal(key_eta, shape=(batch_size, n_buses))
            * np.sqrt(_LOG_NORMAL_SIGMA_SQ)
            + _LOG_NORMAL_MU
        )
        eta = jnp.exp(z)
        return (gammas[:, None] * eta * d_ref[None, :]).astype(d_ref.dtype)

    def make_splits(key: jax.Array, batch_size: int):
        d_batch = sample_d_batch(key, batch_size)
        p_max_batch = jnp.broadcast_to(p_max, (batch_size, n_gen))
        if mode == "ed":
            r_max_batch = jnp.zeros((batch_size, n_gen), dtype=p_max.dtype)
            R_batch = jnp.zeros(batch_size)
        else:
            p_max_sum = jnp.sum(p_max)
            p_max_inf = jnp.max(p_max)
            alpha_r = 5.0 * p_max_inf / jnp.maximum(p_max_sum, 1e-12)
            r_max_batch = jnp.broadcast_to(
                (alpha_r * p_max).astype(p_max.dtype), (batch_size, n_gen)
            )
            R_batch = (
                jr.uniform(
                    jr.fold_in(key, 2),
                    shape=(batch_size,),
                    minval=1.0,
                    maxval=2.0,
                )
                * p_max_inf
            )
        return d_batch, p_max_batch, r_max_batch, R_batch

    return {
        "train": make_splits(key_train, train_size),
        "val": make_splits(key_val, val_size),
        "test": make_splits(key_test, test_size),
    }


def load_reference_case(path: str | Path) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load reference case from npz with keys 'd_ref' and 'p_max'.

    Optional keys: 'n_buses', 'n_generators' (for validation).
    PGLib cases (e.g. ieee300, pegase1k) can be exported to this format
    externally (e.g. from MATPOWER/PGLib in MATLAB or a separate script).
    """
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(
            "Only .npz format is supported; file must have keys 'd_ref' and 'p_max'."
        )
    data = np.load(path)
    if "d_ref" not in data or "p_max" not in data:
        raise KeyError("npz must contain 'd_ref' and 'p_max'.")
    d_ref = jnp.array(data["d_ref"])
    p_max = jnp.array(data["p_max"])
    return d_ref, p_max


def make_synthetic_reference(
    n_buses: int = 10,
    n_generators: int = 5,
    *,
    key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a minimal synthetic reference case for testing.

    d_ref and p_max are positive random arrays. Not from PGLib; for unit tests
    or running the pipeline without external data.
    """
    if key is None:
        key = jr.PRNGKey(0)
    k1, k2 = jr.split(key)
    d_ref = jnp.maximum(jr.uniform(k1, (n_buses,), minval=0.1, maxval=2.0), 0.01)
    p_max = jnp.maximum(jr.uniform(k2, (n_generators,), minval=0.5, maxval=5.0), 0.1)
    return d_ref, p_max


def save_dataset(
    dataset: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    out_dir: str | Path,
) -> None:
    """Save train/val/test arrays to out_dir as npz per split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (d, p_max, r_max, R) in dataset.items():
        np.savez(
            out_dir / f"{split_name}.npz",
            d=np.asarray(d),
            p_max=np.asarray(p_max),
            r_max=np.asarray(r_max),
            R=np.asarray(R),
        )


def load_dataset(
    out_dir: str | Path,
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Load train/val/test from out_dir (npz per split)."""
    out_dir = Path(out_dir)
    result = {}
    for split_name in ("train", "val", "test"):
        p = out_dir / f"{split_name}.npz"
        if not p.exists():
            continue
        data = np.load(p)
        result[split_name] = (
            jnp.array(data["d"]),
            jnp.array(data["p_max"]),
            jnp.array(data["r_max"]),
            jnp.array(data["R"]),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate E2ELR datasets (paper Sec. 5.1). Self-supervised only; no labels."
    )
    parser.add_argument(
        "--ref_case",
        type=str,
        required=True,
        help="Path to reference case .npz with keys 'd_ref' and 'p_max'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ed", "edr"],
        default="edr",
        help="ED (no reserve) or ED-R (with reserve).",
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=50_000,
        help="Total instances (default 50000); split 40000 train, 5000 val, 5000 test.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Output directory for train/val/test .npz files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    train_size = 40_000
    val_size = 5_000
    test_size = 5_000
    if args.n_instances != train_size + val_size + test_size:
        parser.error(
            f"--n_instances must be {train_size + val_size + test_size} (40000+5000+5000) for paper split."
        )

    key = jr.PRNGKey(args.seed)
    d_ref, p_max = load_reference_case(args.ref_case)
    dataset = generate_dataset_vectorized(
        key,
        d_ref,
        p_max,
        n_total=args.n_instances,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        mode=args.mode,
    )
    save_dataset(dataset, args.out_dir)
    print(
        f"Saved {args.mode.upper()} dataset to {args.out_dir}: "
        f"train {dataset['train'][0].shape[0]}, val {dataset['val'][0].shape[0]}, test {dataset['test'][0].shape[0]}."
    )


if __name__ == "__main__":
    main()
