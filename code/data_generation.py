"""
Data generation for E2ELR (paper Sec. 5.1).

Generates ED and ED-R instances by perturbing a reference load profile.
Training is self-supervised only; no labels (optimal solutions) are generated.

Includes a Grain-compatible on-demand data source: instances are generated
from an index-derived key (no storage, consistent across epochs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, SupportsIndex

import jax
import jax.numpy as jnp
import jax.random as jr
from grain.python import RandomAccessDataSource
import grain
import numpy as np


# --- Grain on-demand data source ----------------------------------------------


class EDInstanceDataSource(RandomAccessDataSource):
    """
    Grain-compatible data source that generates ED/ED-R instances on demand.

    Does not store generated data: each record is produced from a deterministic
    key derived from its index (key = fold_in(seed, index)), so the same index
    always yields the same instance across epochs and runs.

    Implements the RandomAccessDataSource protocol: __len__ and __getitem__(int).
    Use with Grain via grain.MapDataset.source(instance) or grain.load(...).
    """

    def __init__(
        self,
        d_ref: jnp.ndarray,
        p_max: jnp.ndarray,
        num_records: int,
        *,
        mode: Literal["ed", "edr"] = "edr",
        seed: int = 0,
    ):
        self._d_ref = d_ref
        self._p_max = p_max
        self._num_records = num_records
        self._mode = mode
        self._seed = seed

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, index: SupportsIndex) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
        idx = int(index)
        if idx < 0 or idx >= self._num_records:
            raise IndexError(
                f"index {idx} out of range [0, {self._num_records})"
            )
        key = jr.fold_in(jr.PRNGKey(self._seed), idx)
        if self._mode == "ed":
            return sample_ed_instance(key, self._d_ref, self._p_max)
        return sample_edr_instance(key, self._d_ref, self._p_max)


def make_grain_dataset(
    d_ref: jnp.ndarray,
    p_max: jnp.ndarray,
    num_records: int,
    *,
    mode: Literal["ed", "edr"] = "edr",
    seed: int = 0,
    shuffle_seed: int | None = 42,
    batch_size: int | None = None,
):
    """
    Build a Grain MapDataset from the on-demand ED instance source.

    Requires: pip install grain

    Args:
        d_ref: Reference demand (n_buses,).
        p_max: Generator max power (n_generators,).
        num_records: Number of virtual records (same index → same instance).
        mode: "ed" or "edr".
        seed: Base RNG seed for index→key (epoch-consistent instances).
        shuffle_seed: If set, shuffle dataset with this seed; None = no shuffle.
        batch_size: If set, batch the dataset; None = return single records.

    Returns:
        A Grain MapDataset (or batched MapDataset). Iterate with:
            ds = make_grain_dataset(...)
            it = iter(ds)
            batch = next(it)  # (d, p_max, r_max, R) or batched
    """
    source = EDInstanceDataSource(
        d_ref, p_max, num_records, mode=mode, seed=seed
    )
    ds = grain.MapDataset.source(source)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if batch_size is not None:
        ds = ds.batch(batch_size=batch_size)
    return ds


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


def load_reference_case(
    path: str | Path,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    dict,
]:
    """
    Load reference case from npz with keys 'd_ref' and 'p_max'.

    Optional keys (for SSL loss with cost and thermal violations): 'c_linear',
    'f_min', 'f_max', 'Phi', 'PTDF_bus', 'Mth'. When present, they are returned
    in a dict as the third element.

    Returns:
        d_ref: (n_buses,) reference demand.
        p_max: (n_generators,) max generation.
        ref_optional: None, or dict with keys c_linear, f_min, f_max, Phi,
            PTDF_bus, Mth (JAX arrays / scalars) when present in the npz.
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

    ref = {
        "c_linear": jnp.array(data["c_linear"]),
        "f_min": jnp.array(data["f_min"]),
        "f_max": jnp.array(data["f_max"]),
        "Phi": jnp.array(data["Phi"]),
        "PTDF_bus": jnp.array(data["PTDF_bus"]),
        "Mth": float(jnp.array(data["Mth"]).item()),
    }

    return d_ref, p_max, ref


if __name__ == "__main__":
    d_ref, p_max, _ = load_reference_case("data/pglib_opf_case30000_goc_ref.npz")
    datasource = EDInstanceDataSource(d_ref, p_max, 100000)
    dataset = grain.MapDataset.source(datasource).batch(64)
    
    for d, p_max, r_max, R in dataset:
        print(d.shape, p_max.shape, r_max.shape, R)
        break
