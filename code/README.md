# E2ELR: End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch

Implementation of models and experiments and my extension of [arXiv:2304.11726v2](https://arxiv.org/abs/2304.11726v2).

## Overview

Trains neural network proxies that approximate solutions to the Economic Dispatch (ED) and Economic Dispatch with Reserves (EDR) problems, while guaranteeing feasibility via differentiable repair layers.

**Models:**

| Model | Description |
|-------|-------------|
| `dnn` | Vanilla fully-connected network (baseline, no feasibility layers) |
| `e2elr` | End-to-End Learning and Repair — DNN + power balance repair + reserve repair |
| `e2elrdc` | E2ELR + DC power flow repair (enforces power flow equations) |

**Problem types:**

| Problem | Description |
|---------|-------------|
| `ed` | Economic Dispatch (no reserves) |
| `edr` | Economic Dispatch with Reserves |

**Training modes:**

| Mode | Description |
|------|-------------|
| `sl` | Supervised learning (minimize MSE to solver solution) |
| `ssl` | Self-supervised learning (minimize cost + constraint penalties directly) |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train a single model

```bash
python main.py \
    --case data/pglib_opf_case300_ieee.m \
    --model e2elr \
    --problem ed \
    --mode ssl
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--case` | `data/pglib_opf_case300_ieee.m` | Path to MATPOWER `.m` case file |
| `--model` | `e2elr` | Model architecture: `dnn`, `e2elr`, `e2elrdc` |
| `--problem` | `ed` | Problem type: `ed` or `edr` |
| `--mode` | `ssl` | Training mode: `sl` or `ssl` |
| `--n_instances` | `50000` | Number of instances to generate |
| `--n_layers` | `3` | Number of hidden layers |
| `--hidden_dim` | `256` | Hidden layer dimension |
| `--lr` | `1e-2` | Initial learning rate |
| `--batch_size` | `64` | Training batch size |
| `--max_epochs` | `500` | Maximum epochs |
| `--patience` | `20` | Early-stopping patience |
| `--max_time` | `150.0` | Max training time (minutes) |
| `--seed` | `42` | Random seed |

### Hyperparameter search

```bash
python hparam_search.py
```

Uses [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) with Optuna. Results are saved under `ray_results/`.

### Run all combinations

```bash
python run_all.py
```

Trains all combinations of `{dnn, e2elr, e2elrdc}` × `{ed, edr}` × `{ieee300, pegase1354}`.

## File Structure

```
code/
├── main.py          # Entry point: data loading, training, evaluation
├── models.py        # DNN, E2ELR, E2ELRDCModel architectures + repair layers
├── train.py         # Dataset construction, training loop, evaluation
├── data_utils.py    # MATPOWER parsing, instance generation, solver interface
├── hparam_search.py # Hyperparameter tuning with Ray Tune + Optuna
├── run_all.py       # Batch runner for all model/problem/case combinations
├── requirements.txt # Python dependencies
├── data/            # MATPOWER .m case files
├── checkpoints/     # Precomputed ground-truth solution caches (.npz)
└── models/          # Saved model weights (.eqx)
```