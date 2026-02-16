# E2ELR: End-to-End Feasible Optimization Proxies for Economic Dispatch

JAX + Equinox implementation of the **E2ELR** architecture from the paper:

**"End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch"**  
W. Chen, M. Tanneau, P. Van Hentenryck (IEEE Trans. Power Systems).

Only the E2ELR network is implemented (no baselines: DNN, DeepOPF, DC3, LOOP).

## Architecture

- **Input:** Nodal demand vector `d`; instance parameters `p_max`, `r_max`, `R` (used in repair layers).
- **DNN backbone:** Fully connected layers (configurable depth `l` ∈ {3,4,5}, hidden size `hd` ∈ {128,256}), ReLU, LayerNorm, Dropout(0.2) after each hidden layer; final layer → sigmoid → `z`; then `p_tilde = z * p_max`.
- **Power balance repair** (Section 3.2, Eq. 3–4): Ensures `e'@p = e'@d` and `0 ≤ p ≤ p_max`.
- **Reserve repair** (Algorithm 1, Section 3.3): Ensures `sum_g min(r_max_g, p_max_g - p_g) ≥ R` while keeping power balance.
- **Output:** Feasible dispatch `p_hat`. Reserves can be recovered as `r_g = min(r_max_g, p_max_g - p_hat_g)` via `repair_layers.reserve_recovery`.

## Setup

```bash
pip install -r requirements.txt
```

Requires `jax` and `equinox`. For GPU, install the appropriate `jaxlib` (e.g. `jax[cuda12]`).

## Usage

```python
import jax
import jax.numpy as jnp
from model import E2ELR, e2elr_batched
from repair_layers import power_balance_repair, reserve_repair, reserve_recovery
from types import EDInstance

# Sizes
n_buses = 100   # demand dimension
n_gen = 50      # number of generators

key = jax.random.PRNGKey(0)
model = E2ELR(
    in_size=n_buses,
    out_size=n_gen,
    num_layers=3,
    hidden_size=256,
    dropout_rate=0.2,
    key=key,
)

# Single instance
d = jnp.ones(n_buses) * 0.5
p_max = jnp.ones(n_gen) * 2.0
r_max = jnp.ones(n_gen) * 0.5
R = 10.0
p_hat = model(d, p_max, r_max, R)
r = reserve_recovery(p_hat, p_max, r_max)

# Batched
d_batch = jnp.ones((32, n_buses)) * 0.5
p_max_batch = jnp.broadcast_to(p_max, (32, n_gen))
r_max_batch = jnp.broadcast_to(r_max, (32, n_gen))
R_batch = jnp.full(32, R)
p_hat_batch = e2elr_batched(model, d_batch, p_max_batch, r_max_batch, R_batch)
```

## Files

| File | Purpose |
|------|--------|
| `repair_layers.py` | Power balance and reserve repair (pure JAX). |
| `model.py` | E2ELR Equinox module (DNN + repair layers). |
| `types.py` | Optional `EDInstance` for (d, p_max, r_max, R). |
| `requirements.txt` | `jax`, `equinox`. |

## Notes

- The paper uses **BatchNorm** after each dense layer; this implementation uses **LayerNorm** for a stateless, JAX-friendly backbone. For exact BatchNorm (with running stats), you would need Equinox’s stateful BatchNorm and to thread state through the forward pass.
- Repair layers are differentiable almost everywhere; gradients at boundaries may be subgradients.
- For training, use `inference=False` and pass a `key` when calling the model so Dropout is applied.

## References

- Power balance: Eq. (3)–(4) and Theorem 1, Section 3.2.
- Reserve: Algorithm 1 and Lemma 1 / Theorem 2, Section 3.3.
- End-to-end architecture: Section 3.4 and Figure (FFR_v2.png).
- Hyperparameters: appendix (depth `l`, hidden `hd`, Dropout 0.2).
