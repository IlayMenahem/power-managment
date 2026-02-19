"""
Profile a full pipeline run on pglib_opf_case1354_pegase.m.

Compares:
  - PTDF:  sparse LU  vs  dense inverse
  - LP:    SciPy HiGHS  vs  parametric CVXPY  vs  original CVXPY
  - Parallel HiGHS with multiprocessing

Usage:
    python profile_run.py                         # default: 100 instances
    python profile_run.py --n_instances 500        # more instances
    python profile_run.py --full_cprofile          # detailed cProfile output
"""

import argparse
import cProfile
import io
import os
import pstats
import time

import numpy as np

from data_utils import (
    parse_matpower,
    extract_case_data,
    compute_ptdf,
    compute_ptdf_dense,
    generate_instances,
    solve_all_instances,
    solve_ed_highs,
    solve_ed_instance,
)


CASE_PATH = "data/pglib_opf_case1354_pegase.m"


def _banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def profile_parse(case_path):
    _banner("1. Parse MATPOWER file")
    t0 = time.perf_counter()
    raw = parse_matpower(case_path)
    case = extract_case_data(raw)
    dt = time.perf_counter() - t0
    print(f"  Time:     {dt:.4f}s")
    print(f"  Buses:    {case['n_bus']}")
    print(f"  Gens:     {case['n_gen']}")
    print(f"  Branches: {case['n_branch']}")
    return raw, case, dt


def profile_ptdf(case):
    _banner("2. PTDF Computation")

    # Sparse LU (new fast path)
    t0 = time.perf_counter()
    ptdf_full, ptdf_gen = compute_ptdf(case)
    dt_sparse = time.perf_counter() - t0
    print(f"  Sparse LU:     {dt_sparse:.4f}s  (shape {ptdf_full.shape})")

    # Dense inverse (original)
    t0 = time.perf_counter()
    ptdf_full_d, ptdf_gen_d = compute_ptdf_dense(case)
    dt_dense = time.perf_counter() - t0
    print(f"  Dense inverse: {dt_dense:.4f}s")

    # Verify equivalence
    diff = np.max(np.abs(ptdf_full - ptdf_full_d))
    print(f"  Max diff:      {diff:.2e}  (should be ~0)")
    if dt_dense > 0:
        print(f"  Speedup:       {dt_dense / dt_sparse:.1f}x")

    return ptdf_full, ptdf_gen, dt_sparse, dt_dense


def profile_instance_gen(case, n_instances):
    _banner("3. Instance Generation")
    t0 = time.perf_counter()
    instances = generate_instances(case, n_instances, problem_type="ed", seed=42)
    dt = time.perf_counter() - t0
    print(f"  Generated {n_instances} instances in {dt:.4f}s")
    return instances, dt


def profile_single_lp(instances, case, ptdf_full, ptdf_gen):
    """Compare a single LP solve across solvers."""
    _banner("4. Single LP Solve (first instance)")
    pd_vec = instances["pd"][0]

    # HiGHS
    t0 = time.perf_counter()
    pg_h, obj_h = solve_ed_highs(pd_vec, case, ptdf_full, ptdf_gen)
    dt_highs = time.perf_counter() - t0

    # CVXPY (original)
    t0 = time.perf_counter()
    pg_c, obj_c = solve_ed_instance(pd_vec, case, ptdf_full, ptdf_gen)
    dt_cvxpy = time.perf_counter() - t0

    print(f"  HiGHS:   {dt_highs:.4f}s  (obj={obj_h:.4f})")
    print(f"  CVXPY:   {dt_cvxpy:.4f}s  (obj={obj_c:.4f})")
    if obj_h is not None and obj_c is not None:
        print(f"  Obj diff: {abs(obj_h - obj_c):.2e}")
    if dt_highs > 0:
        print(f"  Speedup:  {dt_cvxpy / dt_highs:.1f}x")

    return dt_highs, dt_cvxpy


def profile_batch_lp(instances, case, ptdf_full, ptdf_gen, n_instances):
    """Compare batch solving across solver backends."""
    _banner(f"5. Batch LP Solve ({n_instances} instances)")

    results = {}

    # (a) HiGHS sequential
    t0 = time.perf_counter()
    pg_h, obj_h = solve_all_instances(
        instances, case, ptdf_full, ptdf_gen,
        solver="highs", n_workers=1, verbose=False,
    )
    dt = time.perf_counter() - t0
    valid_h = int(np.sum(~np.isnan(obj_h)))
    results["highs_seq"] = dt
    print(f"  HiGHS sequential:    {dt:.2f}s  "
          f"({dt / n_instances:.4f}s/inst, {valid_h} valid)")

    # (b) HiGHS parallel
    n_cores = min(os.cpu_count() or 1, 8)
    if n_cores > 1:
        t0 = time.perf_counter()
        pg_hp, obj_hp = solve_all_instances(
            instances, case, ptdf_full, ptdf_gen,
            solver="highs", n_workers=n_cores, verbose=False,
        )
        dt = time.perf_counter() - t0
        results["highs_par"] = dt
        print(f"  HiGHS parallel ({n_cores}w): {dt:.2f}s  "
              f"({dt / n_instances:.4f}s/inst)")

    # (c) Parametric CVXPY
    t0 = time.perf_counter()
    pg_p, obj_p = solve_all_instances(
        instances, case, ptdf_full, ptdf_gen,
        solver="cvxpy_param", verbose=False,
    )
    dt = time.perf_counter() - t0
    results["cvxpy_param"] = dt
    print(f"  CVXPY parametric:    {dt:.2f}s  "
          f"({dt / n_instances:.4f}s/inst)")

    # (d) Original CVXPY
    t0 = time.perf_counter()
    pg_c, obj_c = solve_all_instances(
        instances, case, ptdf_full, ptdf_gen,
        solver="cvxpy", verbose=False,
    )
    dt = time.perf_counter() - t0
    results["cvxpy_orig"] = dt
    print(f"  CVXPY original:      {dt:.2f}s  "
          f"({dt / n_instances:.4f}s/inst)")

    # Verify all solvers agree
    mask = ~np.isnan(obj_h) & ~np.isnan(obj_c)
    if mask.any():
        max_diff = np.max(np.abs(obj_h[mask] - obj_c[mask]))
        print(f"\n  Max obj diff (HiGHS vs CVXPY): {max_diff:.2e}")

    return results


def print_summary(timings, n_instances):
    _banner("SUMMARY")
    total_orig = (timings["parse"] + timings["ptdf_dense"]
                  + timings["gen"] + timings["cvxpy_orig"])
    total_fast = (timings["parse"] + timings["ptdf_sparse"]
                  + timings["gen"] + timings["highs_seq"])

    print(f"  {'Component':<25} {'Original':>10} {'Optimized':>10} {'Speedup':>8}")
    print(f"  {'─' * 55}")
    print(f"  {'Parse':<25} {timings['parse']:>10.3f}s {timings['parse']:>10.3f}s {'1.0x':>8}")
    print(f"  {'PTDF':<25} {timings['ptdf_dense']:>10.3f}s {timings['ptdf_sparse']:>10.3f}s "
          f"{timings['ptdf_dense'] / max(timings['ptdf_sparse'], 1e-9):>7.1f}x")
    print(f"  {'Instance gen':<25} {timings['gen']:>10.3f}s {timings['gen']:>10.3f}s {'1.0x':>8}")
    print(f"  {'LP solve (sequential)':<25} {timings['cvxpy_orig']:>10.2f}s {timings['highs_seq']:>10.2f}s "
          f"{timings['cvxpy_orig'] / max(timings['highs_seq'], 1e-9):>7.1f}x")

    if "highs_par" in timings:
        print(f"  {'LP solve (parallel)':<25} {'—':>10} {timings['highs_par']:>10.2f}s "
              f"{timings['cvxpy_orig'] / max(timings['highs_par'], 1e-9):>7.1f}x")

    print(f"  {'─' * 55}")
    print(f"  {'TOTAL (sequential)':<25} {total_orig:>10.2f}s {total_fast:>10.2f}s "
          f"{total_orig / max(total_fast, 1e-9):>7.1f}x")

    if "highs_par" in timings:
        total_par = (timings["parse"] + timings["ptdf_sparse"]
                     + timings["gen"] + timings["highs_par"])
        print(f"  {'TOTAL (parallel)':<25} {total_orig:>10.2f}s {total_par:>10.2f}s "
              f"{total_orig / max(total_par, 1e-9):>7.1f}x")

    # Extrapolate to 5000 instances
    scale = 5000 / n_instances
    ext_orig = timings["parse"] + timings["ptdf_dense"] + timings["gen"] + timings["cvxpy_orig"] * scale
    ext_fast = timings["parse"] + timings["ptdf_sparse"] + timings["gen"] + timings["highs_seq"] * scale
    print("\n  Extrapolated to 5000 instances:")
    print(f"    Original: ~{ext_orig / 60:.1f} min")
    print(f"    Optimized (seq): ~{ext_fast / 60:.1f} min")
    if "highs_par" in timings:
        ext_par = timings["parse"] + timings["ptdf_sparse"] + timings["gen"] + timings["highs_par"] * scale
        print(f"    Optimized (par): ~{ext_par / 60:.1f} min")


def run_profile(n_instances=100):
    """Run the full profiling pipeline."""
    timings = {}

    raw, case, dt = profile_parse(CASE_PATH)
    timings["parse"] = dt

    ptdf_full, ptdf_gen, dt_sparse, dt_dense = profile_ptdf(case)
    timings["ptdf_sparse"] = dt_sparse
    timings["ptdf_dense"] = dt_dense

    instances, dt = profile_instance_gen(case, n_instances)
    timings["gen"] = dt

    profile_single_lp(instances, case, ptdf_full, ptdf_gen)

    batch_results = profile_batch_lp(instances, case, ptdf_full, ptdf_gen, n_instances)
    timings.update(batch_results)

    print_summary(timings, n_instances)


def run_cprofile(n_instances=100):
    """Run with cProfile for detailed function-level analysis."""
    pr = cProfile.Profile()
    pr.enable()
    run_profile(n_instances)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    _banner("cProfile — Top 40 by cumulative time")
    print(s.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile E2ELR data pipeline")
    parser.add_argument("--n_instances", type=int, default=100,
                        help="Number of instances to profile (default: 100)")
    parser.add_argument("--full_cprofile", action="store_true",
                        help="Include detailed cProfile output")
    args = parser.parse_args()

    if args.full_cprofile:
        run_cprofile(args.n_instances)
    else:
        run_profile(args.n_instances)
