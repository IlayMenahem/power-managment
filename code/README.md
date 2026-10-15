# E2ELR — MLX version

MLX port of `../code/` (PyTorch), changed to follow the paper more closely (see below).
Models `dnn` and `e2elr`, problems `ed` and `edr`, training modes `sl` and `ssl`, and
thermal limits `soft` (paper) or `hard`.

```bash
pip install -r requirements.txt   # needs Python >= 3.10
python main.py --case data/pglib_opf_case300_ieee.m --model e2elr --problem edr --mode ssl
python data_utils.py --case data/pglib_opf_case300_ieee.m --problem edr --line_limits hard
python train_all.py               # the paper's training settings, no search
```

`data/` and `checkpoints/` are symlinks to `../code/`.

## Hard thermal limits
About 20% of the paper's ED-R instances have no solution when thermal limits are hard
(they are the high-load ones). `--line_limits hard` builds a dataset by rejection
sampling: a pool of `ceil(1.35 n)` instances is generated as in the paper, each is solved
with hard limits, and the first `n` feasible ones are kept. Checkpoints store the kept
indices, not the loads, so `gt_{case}_{problem}_{soft|hard}_n{n}_s{seed}.npz` is small.

## Precision
The backbone (Linear/BatchNorm/ReLU/Dropout) runs in float32 on the GPU. Repair layers,
DC power flow, violations and losses are float64, which MLX only supports on the CPU
stream, so `models.py` sets the default device to CPU and runs the backbone under
`mx.stream(mx.gpu)`. Memory is unified, so there are no copies between the two.

## Differences from the PyTorch version
Modelling (closer to the paper):
- The network predicts only `pg`. The angles solve the DC power flow equations for `pg`
  (`theta_ref = 0`, sparse LU of `B_r`), so branch flows equal the paper's PTDF flows.
  There is no theta head, no `e2elrdc` projection and no DC penalty.
- Phase shift angles are ignored, as in the paper's flows `Phi (pg - pd)` (Eq. 1g).
- Generator minimums `Pmin` are kept; the repair layers act on `pg - pg_min` (the paper's
  "without loss of generality, Pmin = 0" shift).
- The network input is `(pd, R)`, the part of the paper's `x` that varies between
  instances.
- The thermal penalty is part of the SSL objective and is not scaled by `lambda`;
  `lambda = 0` for `e2elr` with SSL, as for the paper's feasible models.
- The DC residual check is per bus (max), not an L1 sum.

Training and reporting (the paper's):
- Adam, lr 1e-2, weight decay 1e-6, batch size 64, no gradient clipping, 3 layers of 256
  units (one point of the paper's grid; there is no hyperparameter search).
- Per hard constraint, % feasible and the shifted geometric mean (1 p.u.) of the
  violations of infeasible predictions (Table III).
- `benchmarks.py`: inference time per batch of 256 (Tables VI-VII), and the repair
  layers vs a Euclidean projection QP solved with HiGHS on one thread (Table X; the
  paper uses Gurobi).

Implementation:
- No `hparam_search.py` / `run_all.py` (Ray Tune).
- `DataLoader` is replaced by `Dataset.batches`; ReduceLROnPlateau is `PlateauScheduler`;
  Adam's L2 weight decay is added to the gradient, as in torch.
- The SL loss uses a masked mean instead of boolean indexing.
- Ground-truth files have new names, so the PyTorch ones in `checkpoints/` are untouched.
