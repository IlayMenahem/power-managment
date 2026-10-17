"""
Time breakdown of the proxy's inference (model.dispatch) per batch of 256
instances, for E2ELR and E2ELR-L on the hard thermal limit ED-R test sets.
Each stage is evaluated on its own (mx.eval) and timed over many repeats; the
weights are untrained, which does not change the time of any stage.

Run from code/:
  uv run --python 3.12 --with mlx --with numpy --with scipy python profile_inference.py
"""

import time

import mlx.core as mx
import numpy as np
from data_utils import load_case, load_dataset
from models import DispatchProxy, line_flow_step
from train import build_datasets

CASES = ("data/pglib_opf_case300_ieee.m", "data/pglib_opf_case1354_pegase.m")
BATCH, N_BATCHES, N_REPEATS = 256, 10, 20


def median_ms(fn, inputs) -> float:
    """Median time of mx.eval(fn(*args)) over the batches and repeats."""
    for args in inputs[:2]:  # warm-up
        mx.eval(fn(*args))
    times = []
    for args in inputs:
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            mx.eval(fn(*args))
            times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


def profile(case_path: str) -> None:
    case = load_case(case_path)
    instances, _ = load_dataset(
        case, "edr", 50000, 42, "hard", "checkpoints", solve_if_missing=False
    )
    test = build_datasets(instances, None)["test"]
    with mx.stream(mx.cpu):
        batches = [
            tuple(t.astype(mx.float64) for t in b[:2])
            for b, _ in zip(test.batches(BATCH), range(N_BATCHES))
        ]
        mx.eval(batches)
    print(f"\n{case.name}: n_bus={case.n_bus}, n_gen={case.n_gen}, n_branch={case.n_branch}")

    for arch in ("e2elr", "e2elr_line"):
        mx.random.seed(0)
        model = DispatchProxy(arch, case, instances.r_max, hidden_dim=256, n_layers=3)
        model.eval()
        with mx.stream(mx.cpu):
            stage_in = []  # (pd, R, pg, anchor, repaired pg, repaired anchor)
            for pd, R in batches:
                pg, anchor = model.initial_dispatch(pd, R)
                pg_r = model.e2elr_repair(pg, pd, R)
                a_r = None if anchor is None else model.e2elr_repair(anchor, pd, R)
                mx.eval(pg, pg_r, *([] if anchor is None else [anchor, a_r]))
                stage_in.append((pd, R, pg, anchor, pg_r, a_r))

            stages = {
                "total (dispatch)": lambda pd, R, *_: model.dispatch(pd, R),
                "backbone": lambda pd, R, *_: model.initial_dispatch(pd, R),
                "E2ELR repairs, dispatch": lambda pd, R, pg, *_: model.e2elr_repair(pg, pd, R),
            }
            if model.has_line_repair:
                Pg, Pd, rate = model._gen_ptdf_t, model._ptdf_t, model._rate
                stages |= {
                    "E2ELR repairs, anchor": lambda pd, R, pg, a, *_: model.e2elr_repair(a, pd, R),
                    "line repair (all)": lambda pd, R, pg, a, pg_r, a_r: model.line_repair(pg_r, a_r, pd),
                    "  pd @ Phi^T": lambda pd, *_: pd @ Pd,
                    "  pg @ Phi_g^T (x2)": lambda pd, R, pg, a, pg_r, a_r: (pg_r @ Pg, a_r @ Pg),
                    "  step + combination": lambda pd, R, pg, a, pg_r, a_r: (
                        lambda f, fa: pg_r + line_flow_step(f, fa, rate) * (a_r - pg_r)
                    )(*flows[id(pd)]),
                }
                flows = {}
                for pd, R, pg, a, pg_r, a_r in stage_in:
                    d = pd @ Pd
                    flows[id(pd)] = (pg_r @ Pg - d, a_r @ Pg - d)
                    mx.eval(*flows[id(pd)])
            print(f"  {arch}")
            for name, fn in stages.items():
                print(f"    {name:<26}{median_ms(fn, stage_in):8.3f} ms")


if __name__ == "__main__":
    for path in CASES:
        profile(path)
