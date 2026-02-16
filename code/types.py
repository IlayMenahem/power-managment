"""
Optional data structures for E2ELR instance data.

Use for clarity when passing (d, p_max, r_max, R) around.
"""

from typing import NamedTuple

import jax.numpy as jnp


class EDInstance(NamedTuple):
    """Single economic dispatch instance (demand + params for repair layers)."""

    d: jnp.ndarray      # Nodal demand (e.g. shape (n_buses,) or (n_loads,))
    p_max: jnp.ndarray  # Max generation per generator (n_generators,)
    r_max: jnp.ndarray  # Max reserve per generator (n_generators,)
    R: float            # Minimum total reserve requirement
