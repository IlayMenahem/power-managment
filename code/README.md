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
| `sl` | Supervised learning (minimize MAE to solver solution + constraint penalties) |
| `ssl` | Self-supervised learning (minimize cost + constraint penalties directly) |

## Installation

```bash
pip install -r requirements.txt
```

Physical quantities (repairs, DC projection, violations) are computed in float64;
the network backbone runs in float32. Use `--device cpu` or `cuda` (`auto` picks
CUDA if available); MPS is not supported because it lacks float64.

## Usage

### Generate ground truth

```bash
python data_utils.py --case data/pglib_opf_case300_ieee.m --problem edr
```

Solves every instance with HiGHS and saves `checkpoints/gt_<case>_<problem>_n<N>_s<seed>.npz`,
which `main.py` loads to compute optimality gaps (and needs for `--mode sl`).

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
| `--skip_solve` | off | Don't solve the LPs when no ground-truth checkpoint exists |
| `--results_file` | none | Save test metrics as JSON |

### Hyperparameter search

```bash
python hparam_search.py
```

Uses [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) with Optuna. Results are saved under `ray_results/`.

### Run all combinations

```bash
python run_all.py    # hyperparameter search for every combination
python train_all.py  # train + evaluate each combination with its best config
```

Both cover `{dnn, e2elr, e2elrdc}` × `{edr}` × `{ieee300, pegase1354}`.
`train_all.py` writes test metrics to `train_results/`.

## File Structure

```
code/
├── main.py          # Entry point: data loading, training, evaluation
├── models.py        # DispatchProxy (dnn / e2elr / e2elrdc) + repair layers
├── train.py         # Dataset construction, training loop, evaluation
├── data_utils.py    # MATPOWER parsing, instance generation, LP ground truth
├── hparam_search.py # Hyperparameter tuning with Ray Tune + Optuna
├── run_all.py       # Hyperparameter search for all combinations
├── train_all.py     # Training with the best configs for all combinations
├── requirements.txt # Python dependencies
├── data/            # MATPOWER .m case files
├── checkpoints/     # Precomputed ground-truth solution caches (.npz)
├── ray_results/     # Best hyperparameter configs
└── train_results/   # Test metrics per combination
```