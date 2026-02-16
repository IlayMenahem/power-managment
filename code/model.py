"""
E2ELR (End-to-End Learning and Repair) model for economic dispatch.

Architecture from "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch": DNN backbone (FC + ReLU + normalization + Dropout, sigmoid,
scale by p_max) -> power balance repair -> reserve repair.

Uses JAX and Equinox. Section 3.4 and Figure (FFR_v2.png) in the paper.
"""

import jax.numpy as jnp
import jax.nn as jnn
from jax import vmap
import equinox as eqx

from repair_layers import power_balance_repair, reserve_repair


class E2ELR(eqx.Module):
    """
    End-to-end feasible optimization proxy.

    Forward: (d, p_max, r_max, R) -> p_hat (feasible dispatch).
    Optionally returns reserves via reserve_recovery(p_hat, p_max, r_max).
    """

    backbone: eqx.nn.MLP
    in_size: int
    out_size: int  # n_generators

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_layers: int = 3,
        hidden_size: int = 256,
        dropout_rate: float = 0.2,
        *,
        key,
    ):
        """
        Args:
            in_size: Dimension of demand input (e.g. number of buses).
            out_size: Number of generators (n).
            num_layers: Number of layers l in {3, 4, 5}; includes output layer.
            hidden_size: Hidden dimension hd in {128, 256}.
            dropout_rate: Unused (kept for API compatibility). eqx.nn.MLP has no dropout.
            key: JAX PRNG key for initialization.
        """
        self.in_size = in_size
        self.out_size = out_size
        self.backbone = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=hidden_size,
            depth=num_layers,
            activation=jnn.relu,
            final_activation=jnn.sigmoid,
            key=key,
        )

    def __call__(
        self,
        d: jnp.ndarray,
        p_max: jnp.ndarray,
        r_max: jnp.ndarray,
        R: float,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        """
        Single instance: d (demand), p_max, r_max, R -> p_hat.
        """
        z = self.backbone(d)
        p_tilde = z * p_max
        p_pb = power_balance_repair(p_tilde, d, p_max)
        p_hat = reserve_repair(p_pb, p_max, r_max, R)
        return p_hat


def e2elr_batched(
    model: E2ELR,
    d_batch: jnp.ndarray,
    p_max_batch: jnp.ndarray,
    r_max_batch: jnp.ndarray,
    R_batch: jnp.ndarray,
    *,
    key=None,
    inference: bool = True,
) -> jnp.ndarray:
    """
    Batched forward: (batch, in_size), (batch, n), (batch, n), (batch,) -> (batch, n).
    """
    def single(d, p_max, r_max, R):
        return model(d, p_max, r_max, R, key=key, inference=inference)

    return vmap(single)(d_batch, p_max_batch, r_max_batch, R_batch)
