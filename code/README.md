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

## Data generation

Data generation follows the paper (Sec. 5.1): instances are obtained by perturbing a reference load profile. **Training is self-supervised only;** no labels (optimal solutions) are generated.

- **Reference case:** An `.npz` file with keys `d_ref` (nodal demand) and `p_max` (generator limits). PGLib cases (e.g. ieee300, pegase1k) can be exported to this format externally (e.g. from MATPOWER/PGLib in MATLAB or a separate script).
- **ED (no reserve):** Perturbed demand \\(d = \\gamma \\eta d^{\\text{ref}}\\) with \\(\\gamma \\sim U[0.8, 1.2]\\) and per-bus \\(\\eta\\) log-normal(mean=1, std=5%); \\(r_{\\max}=0\\), \\(R=0\\).
- **ED-R (with reserve):** Same \\(d\\); \\(\\bar{r}_g = \\alpha_r \\bar{p}_g\\) with \\(\\alpha_r = 5 \\|\\bar{p}\\|_\\infty / \\|\\bar{p}\\|_1\\), and \\(R \\sim U(1,2) \\times \\max_g \\bar{p}_g\\).
- **Splits:** 50,000 instances per case → 40,000 train, 5,000 validation, 5,000 test.

**Run data generation:**

```bash
python data_generation.py --ref_case path/to/ref.npz --mode edr --n_instances 50000 --out_dir data
```

Output: `data/train.npz`, `data/val.npz`, `data/test.npz`, each with arrays `d`, `p_max`, `r_max`, `R` compatible with `E2ELR` and `e2elr_batched`. Use `--mode ed` for ED (no reserve). See `data_generation.py` for programmatic usage (`load_reference_case`, `generate_dataset_vectorized`, `make_synthetic_reference`, `load_dataset`).

**Create dataset from PGLib .m cases (in `data/`):**

If you have PGLib MATPOWER `.m` case files in `code/data/`, use `create_dataset.py` to parse them and build train/val/test splits:

```bash
# Process all .m files in data/
python create_dataset.py --data_dir data --mode edr --n_instances 50000

# Process a single case
python create_dataset.py --data_dir data --case pglib_opf_case300_ieee.m --mode edr --n_instances 50000
```

This writes, per case: `data/<case_stem>_ref.npz` (reference `d_ref`, `p_max`) and `data/<case_stem>/train.npz`, `val.npz`, `test.npz`.

## Files

| File | Purpose |
|------|--------|
| `create_dataset.py` | Build datasets from PGLib .m case files: parse bus/gen → ref .npz, then run data generation per case. |
| `data_generation.py` | Data generation (paper Sec. 5.1): ED/ED-R sampling, dataset builder, reference loader. |
| `repair_layers.py` | Power balance and reserve repair (pure JAX). |
| `model.py` | E2ELR Equinox module (DNN + repair layers). |
| `types.py` | Optional `EDInstance` for (d, p_max, r_max, R). |
| `requirements.txt` | `jax`, `equinox`, `numpy`. |

## Notes

- The paper uses **BatchNorm** after each dense layer; this implementation uses **LayerNorm** for a stateless, JAX-friendly backbone. For exact BatchNorm (with running stats), you would need Equinox’s stateful BatchNorm and to thread state through the forward pass.
- Repair layers are differentiable almost everywhere; gradients at boundaries may be subgradients.
- For training, use `inference=False` and pass a `key` when calling the model so Dropout is applied.

## References

- Power balance: Eq. (3)–(4) and Theorem 1, Section 3.2.
- Reserve: Algorithm 1 and Lemma 1 / Theorem 2, Section 3.3.
- End-to-end architecture: Section 3.4 and Figure (FFR_v2.png).
- Hyperparameters: appendix (depth `l`, hidden `hd`, Dropout 0.2).
