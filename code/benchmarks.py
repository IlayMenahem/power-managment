"""
Computing-time benchmarks of the paper (Section VII-B):
  - inference time of a proxy per batch of 256 instances (Tables VI-VII)
  - repair layers vs the Euclidean projection onto the feasible set, solved as
    a QP with HiGHS on a single thread (Table X; the paper uses Gurobi). For
    e2elr_line the feasible set includes the hard thermal limits.

Reference: "End-to-End Feasible Optimization Proxies for Large-Scale
Economic Dispatch" (arXiv 2304.11726v2), Section VII.
"""

import time

import highspy
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from data_utils import Case, ptdf
from models import f64
from scipy import sparse
from train import Dataset


def inference_time_ms(
    model: nn.Module, dataset: Dataset, batch_size: int = 256, n_warmup: int = 2
) -> float:
    """
    Median time of the proxy's output p_hat for a batch of batch_size instances.
    The angles are not timed: they are not part of the paper's proxy output
    (e2elr_line's line flow repair uses PTDF flows, which are timed).
    """
    model.eval()
    times = []
    for i, batch in enumerate(dataset.batches(batch_size)):
        if len(batch[0]) < batch_size:
            continue
        with mx.stream(mx.cpu):
            pd, R = (t.astype(mx.float64) for t in batch[:2])
            mx.eval(pd, R)
            t0 = time.perf_counter()
            mx.eval(model.dispatch(pd, R))
            elapsed = time.perf_counter() - t0
        if i >= n_warmup:
            times.append(elapsed)
    return 1e3 * float(np.median(times))


class EuclideanProjection:
    """
    min ||p - p0||^2  s.t.  sum(p) = D, sum(r) >= R, p + r <= pg_max,
                            pg_min <= p <= pg_max, 0 <= r <= r_max
                            (line_limits: |Phi (C_g p - pd)| <= rate)
    over x = [p, r], solved with HiGHS (QP) on a single thread. The PTDF rows
    Phi C_g are dense.
    """

    def __init__(self, case: Case, r_max: np.ndarray, line_limits: bool = False):
        n = case.n_gen
        inf = highspy.kHighsInf
        ones = sparse.csr_matrix(np.ones((1, n)))
        eye = sparse.identity(n, format="csr")
        # Rows: 0 power balance, 1 reserve requirement, 2 + g: p_g + r_g <= pg_max_g,
        # then (line_limits) one row per branch. The bounds of the power balance,
        # reserve and branch rows are set per instance in solve().
        blocks = [[ones, None], [None, ones], [eye, eye]]
        self.row_lower = np.concatenate([[0.0, 0.0], np.full(n, -inf)])
        self.row_upper = np.concatenate([[0.0, inf], case.pg_max])
        self.ptdf, self.rate = None, case.rate
        if line_limits:
            self.ptdf = ptdf(case)
            blocks.append([sparse.csr_matrix(self.ptdf[:, case.gen_bus]), None])
            self.row_lower = np.concatenate([self.row_lower, np.full(case.n_branch, -inf)])
            self.row_upper = np.concatenate([self.row_upper, np.full(case.n_branch, inf)])
        A = sparse.bmat(blocks, format="csc")
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = 2 * n, A.shape[0]
        lp.col_lower_ = np.concatenate([case.pg_min, np.zeros(n)])
        lp.col_upper_ = np.concatenate([case.pg_max, r_max])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        gens = np.arange(n)
        hessian = highspy.HighsHessian()  # 2 I on the p block
        hessian.dim_ = 2 * n
        hessian.format_ = highspy.HessianFormat.kTriangular
        hessian.start_ = np.concatenate([np.arange(n + 1), np.full(n, n)])
        hessian.index_ = gens.astype(np.int32)
        hessian.value_ = np.full(n, 2.0)
        self.lp, self.hessian, self.n = lp, hessian, n

    def solve(self, p0: np.ndarray, pd: np.ndarray, R: float) -> tuple[np.ndarray, float]:
        """Projection of p0 and the solver time in seconds (NaNs if not optimal)."""
        self.lp.col_cost_ = np.concatenate([-2.0 * p0, np.zeros(self.n)])
        row_lower, row_upper = self.row_lower.copy(), self.row_upper.copy()
        row_lower[:2] = pd.sum(), R
        row_upper[0] = pd.sum()
        if self.ptdf is not None:  # -rate <= Phi C_g p - Phi pd <= rate
            demand_flow = self.ptdf @ pd
            row_lower[2 + self.n :] = demand_flow - self.rate
            row_upper[2 + self.n :] = demand_flow + self.rate
        self.lp.row_lower_, self.lp.row_upper_ = row_lower, row_upper
        model = highspy.HighsModel()
        model.lp_, model.hessian_ = self.lp, self.hessian
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.passModel(model)
        t0 = time.perf_counter()
        h.run()
        elapsed = time.perf_counter() - t0
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return np.full(self.n, np.nan), elapsed
        return np.array(h.getSolution().col_value[: self.n]), elapsed


def repair_vs_projection(
    model: nn.Module,
    dataset: Dataset,
    case: Case,
    r_max: np.ndarray,
    n_instances: int = 100,
    n_repeats: int = 5,
) -> dict[str, float]:
    """
    Feasibility restoration of the backbone's initial predictions p0 on the
    first n_instances of the dataset, by the repair layers and by the
    Euclidean projection onto the set they enforce (with the hard thermal
    limits for e2elr_line, whose repair also uses the predicted anchor).
    Times are per instance:
      - repair_us: median single-instance time of the repair layers
      - repair_batch_us: repair layers on all n_instances at once, divided by n
      - projection_us: median QP solver time
    Also returns the mean L2 distance ||p - p0|| of both (the projection is,
    by definition, never farther than the repair layers).
    """
    model.eval()
    pd, R = (c[:n_instances].astype(np.float64) for c in dataset.columns[:2])
    with mx.stream(mx.cpu):
        p0, a0 = model.initial_dispatch(f64(pd), f64(R))
        mx.eval(p0)
    p0 = np.array(p0)
    a0 = None if a0 is None else np.array(a0)

    def repair(p, d, r, a):
        with mx.stream(mx.cpu):
            out = model.repair(p, d, r, a)
            mx.eval(out)
        return out

    def anchor_rows(rows: slice) -> mx.array | None:
        return None if a0 is None else f64(a0[rows])

    single, repaired = [], []
    for i in range(len(pd)):
        rows = slice(i, i + 1)
        args = (f64(p0[rows]), f64(pd[rows]), f64(R[rows]), anchor_rows(rows))
        repair(*args)  # warm-up
        runs = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            out = repair(*args)
            runs.append(time.perf_counter() - t0)
        single.append(np.median(runs))
        repaired.append(np.array(out)[0])
    batch_args = (f64(p0), f64(pd), f64(R), anchor_rows(slice(None)))
    repair(*batch_args)  # warm-up
    t0 = time.perf_counter()
    repair(*batch_args)
    batch_time = (time.perf_counter() - t0) / len(pd)

    projection = EuclideanProjection(case, r_max, line_limits=model.has_line_repair)
    projected, proj_times = [], []
    for i in range(len(pd)):
        p, elapsed = projection.solve(p0[i], pd[i], R[i, 0])
        projected.append(p)
        proj_times.append(elapsed)
    projected = np.array(projected)
    solved = ~np.isnan(projected).any(axis=1)

    repair_us = 1e6 * float(np.median(single))
    projection_us = 1e6 * float(np.median(proj_times))
    return {
        "repair_us": repair_us,
        "repair_batch_us": 1e6 * batch_time,
        "projection_us": projection_us,
        "projection_speedup": projection_us / repair_us,
        "projection_failures": int((~solved).sum()),
        "repair_dist_pu": float(np.linalg.norm(np.array(repaired) - p0, axis=1).mean()),
        "projection_dist_pu": float(
            np.linalg.norm(projected[solved] - p0[solved], axis=1).mean()
        ),
    }
