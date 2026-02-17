"""
E2ELR (End-to-End Learning and Repair) model for economic dispatch, Power balance and reserve repair layers for E2ELR.

Implements closed-form, differentiable repair layers from:
  "End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch"
  - Power balance: Eq. (3)-(4) and Theorem 1, Section 3.2
  - Reserve: Algorithm 1 and Lemma 1 / Theorem 2, Section 3.3
Architecture from "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch": DNN backbone (FC + ReLU + normalization + Dropout, sigmoid,
scale by p_max) -> power balance repair -> reserve repair.

Uses JAX and Equinox. Section 3.4 and Figure (FFR_v2.png) in the paper.
"""

import jax.numpy as jnp
import jax.nn as jnn
from jax import vmap, jit
import equinox as eqx


def power_balance_repair(p: jnp.ndarray, d: jnp.ndarray, p_max: jnp.ndarray) -> jnp.ndarray:
    """
    Power balance repair layer P(p).

    Takes dispatch p in [0, p_max] and returns p_tilde with e'@p_tilde = e'@d
    and 0 <= p_tilde <= p_max. Uses proportional scaling (Eq. 3-4).

    Args:
        p: Dispatch vector (n_generators,), in bounds [0, p_max].
        d: Nodal demand vector; D = sum(d) is total demand.
        p_max: Maximum generation (n_generators,).

    Returns:
        p_tilde: Feasible dispatch satisfying power balance and bounds.
    """
    D = jnp.sum(d)
    sum_p = jnp.sum(p)
    sum_p_max = jnp.sum(p_max)

    # eta_up: when sum(p) < D, scale toward p_max
    denom_up = sum_p_max - sum_p
    eta_up_raw = jnp.where(denom_up > 1e-12, (D - sum_p) / denom_up, 0.0)
    eta_up = jnp.clip(eta_up_raw, 0.0, 1.0)

    # eta_dn: when sum(p) >= D, scale toward 0
    eta_dn_raw = jnp.where(sum_p > 1e-12, (sum_p - D) / sum_p, 0.0)
    eta_dn = jnp.clip(eta_dn_raw, 0.0, 1.0)

    p_tilde_up = (1.0 - eta_up) * p + eta_up * p_max
    p_tilde_dn = (1.0 - eta_dn) * p

    # Select branch: if sum(p) < D use up, else use down
    return jnp.where(sum_p < D, p_tilde_up, p_tilde_dn)


def reserve_repair(
    p: jnp.ndarray,
    p_max: jnp.ndarray,
    r_max: jnp.ndarray,
    R: float,
) -> jnp.ndarray:
    """
    Reserve repair layer R(p).

    Takes p in hypersimplex(D) and returns p_tilde in hypersimplex(D) satisfying
    sum_g min(r_max_g, p_max_g - p_g) >= R. Implements Algorithm 1.

    Args:
        p: Dispatch (n_generators,) after power balance repair.
        p_max: Maximum generation (n_generators,).
        r_max: Maximum reserve (n_generators,).
        R: Minimum total reserve requirement (scalar).

    Returns:
        p_tilde: Dispatch satisfying power balance and reserve requirement.
    """
    # r*_g = min(r_max_g, p_max_g - p_g)
    available_reserve = jnp.minimum(r_max, p_max - p)
    total_reserve = jnp.sum(available_reserve)
    delta_R = R - total_reserve

    # p_eco = p_max_g - r_max_g (dispatch level above which reserve is capped by r_max)
    p_eco = p_max - r_max

    # G_up: p_g <= p_eco_g  (can increase dispatch without reducing reserve headroom)
    # G_down: p_g > p_eco_g (must decrease dispatch to free reserve)
    in_G_up = p <= p_eco
    in_G_down = ~in_G_up

    # Delta_up = sum over G_up of (p_eco_g - p_g)
    headroom_up = jnp.maximum(p_eco - p, 0.0)
    delta_up = jnp.sum(jnp.where(in_G_up, headroom_up, 0.0))
    # Delta_down = sum over G_down of (p_g - p_eco_g)
    excess_down = jnp.maximum(p - p_eco, 0.0)
    delta_down = jnp.sum(jnp.where(in_G_down, excess_down, 0.0))

    # Safe division: avoid 0/0; when delta_up or delta_down is 0, no adjustment
    safe_delta_up = jnp.maximum(delta_up, 1e-12)
    safe_delta_down = jnp.maximum(delta_down, 1e-12)

    delta = jnp.maximum(0.0, jnp.minimum(delta_R, jnp.minimum(delta_up, delta_down)))
    alpha_up = jnp.where(delta_up > 1e-12, delta / safe_delta_up, 0.0)
    alpha_down = jnp.where(delta_down > 1e-12, delta / safe_delta_down, 0.0)

    # G_up: move p_g toward p_eco (increase dispatch)
    p_new_up = (1.0 - alpha_up) * p + alpha_up * p_eco
    # G_down: move p_g toward p_eco (decrease dispatch)
    p_new_down = (1.0 - alpha_down) * p + alpha_down * p_eco

    p_tilde = jnp.where(in_G_up, p_new_up, p_new_down)
    return p_tilde


def reserve_recovery(
    p: jnp.ndarray,
    p_max: jnp.ndarray,
    r_max: jnp.ndarray,
) -> jnp.ndarray:
    """
    Recover reserve allocation from dispatch p.

    r*_g = min(r_max_g, p_max_g - p_g) (Eq. 15 in paper).

    Args:
        p: Feasible dispatch (n_generators,).
        p_max: Maximum generation (n_generators,).
        r_max: Maximum reserve (n_generators,).

    Returns:
        r: Reserve allocation (n_generators,).
    """
    return jnp.minimum(r_max, p_max - p)


def thermal_violations_l1(
    p: jnp.ndarray,
    d: jnp.ndarray,
    Phi: jnp.ndarray,
    PTDF_bus: jnp.ndarray,
    f_min: jnp.ndarray,
    f_max: jnp.ndarray,
) -> jnp.ndarray:
    """
    Thermal violation vector and L1 norm for SSL loss (Eq. 32, training.tex).

    flow = Phi @ p - PTDF_bus @ d; xi_th = max(0, flow - f_max) + max(0, f_min - flow).
    Returns sum(xi_th) (scalar) for use in loss = c(p) + Mth * ||xi_th||_1.

    Args:
        p: Dispatch (n_generators,).
        d: Nodal demand (n_buses,).
        Phi: (n_branches, n_generators) = PTDF_bus @ Cg.
        PTDF_bus: (n_branches, n_buses).
        f_min, f_max: (n_branches,) branch thermal limits.

    Returns:
        Scalar: ||xi_th||_1 = sum over branches of thermal violations.
    """
    flow = Phi @ p - PTDF_bus @ d
    xi_above = jnp.maximum(flow - f_max, 0.0)
    xi_below = jnp.maximum(f_min - flow, 0.0)
    xi_th = xi_above + xi_below
    return jnp.sum(xi_th)


# JIT-compiled versions for performance
power_balance_repair_jit = jit(power_balance_repair)
reserve_repair_jit = jit(reserve_repair)
reserve_recovery_jit = jit(reserve_recovery)

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
        *,
        key,
    ):
        """
        Args:
            in_size: Dimension of demand input (e.g. number of buses).
            out_size: Number of generators (n).
            num_layers: Number of layers l in {3, 4, 5}; includes output layer.
            hidden_size: Hidden dimension hd in {128, 256}.
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
