"""
Create E2ELR datasets from PGLib .m case files in code/data.

Parses MATPOWER/PGLib .m files to extract d_ref (nodal demand) and p_max
(generator limits), then uses data_generation to produce train/val/test splits.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np



# MATPOWER bus matrix: bus_i=0, type=1, Pd=2, Qd=3, ...
BUS_PD_COL = 2
# MATPOWER gen matrix: bus=0, Pg=1, Qg=2, Qmax=3, Qmin=4, Vg=5, mBase=6, status=7, Pmax=8, Pmin=9
GEN_BUS_COL = 0
GEN_STATUS_COL = 7
GEN_PMAX_COL = 8


def _parse_matrix_block(lines: list[str], start_idx: int) -> tuple[np.ndarray, int]:
    """
    Parse a MATLAB-style matrix from lines starting at start_idx.
    Returns (array, index of line after the closing '];').
    """
    rows = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        # End of matrix
        if "];" in line:
            # Maybe data on same line: "  val1  val2  ];"
            before_semicolon = line.split("];")[0].strip()
            if before_semicolon:
                parts = re.split(r"\s+", before_semicolon.strip())
                row = [float(x) for x in parts if x]
                if row:
                    rows.append(row)
            i += 1
            break
        # Strip trailing "; % comment" or ";"
        line = re.sub(r";\s*(%.*)?$", "", line).strip()
        if not line:
            i += 1
            continue
        parts = re.split(r"\s+", line)
        row = []
        for x in parts:
            if not x:
                continue
            try:
                row.append(float(x))
            except ValueError:
                break
        if row:
            rows.append(row)
        i += 1

    if not rows:
        return np.array([]).reshape(0, 0), i
    return np.array(rows), i


def _parse_base_mva(path: Path) -> float:
    """Parse mpc.baseMVA from .m file; default 1.0 if not found."""
    text = path.read_text()
    for line in text.splitlines():
        if "baseMVA" in line and "=" in line:
            # e.g. "mpc.baseMVA = 100.0;"
            parts = line.split("=")
            if len(parts) >= 2:
                val = parts[1].strip().rstrip(";").strip()
                try:
                    return float(val)
                except ValueError:
                    pass
    return 1.0


def parse_pglib_m(
    path: str | Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, float,
]:
    """
    Parse a PGLib/MATPOWER .m case file.

    Returns:
        d_ref: (n_buses,) nodal real power demand Pd from mpc.bus
        p_max: (n_generators,) Pmax for each *online* generator from mpc.gen
        branch_matrix: (n_branches, n_cols) from mpc.branch
        gencost_matrix: (n_generators_full, n_cols) from mpc.gencost (full gen order)
        bus_matrix: (n_buses, n_cols) from mpc.bus
        gen_matrix: (n_generators_full, n_cols) from mpc.gen (full, for gen_bus)
        gen_bus: (n_generators,) 0-based bus index for each *online* generator
        base_mva: float from mpc.baseMVA
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    bus_matrix = None
    gen_matrix = None
    branch_matrix = None
    gencost_matrix = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if "mpc.bus" in line and "=" in line and "[" in line:
            i += 1  # move to first data line (opening "mpc.bus = [" is on its own line)
            bus_matrix, i = _parse_matrix_block(lines, i)
            continue
        if "mpc.gen " in line and "=" in line and "[" in line and "mpc.gencost" not in line:
            i += 1
            gen_matrix, i = _parse_matrix_block(lines, i)
            continue
        if "mpc.gencost" in line and "=" in line and "[" in line:
            i += 1
            gencost_matrix, i = _parse_matrix_block(lines, i)
            continue
        if "mpc.branch" in line and "=" in line and "[" in line:
            i += 1
            branch_matrix, i = _parse_matrix_block(lines, i)
            continue
        i += 1

    if bus_matrix is None or bus_matrix.size == 0:
        raise ValueError(f"{path}: could not parse mpc.bus")
    if gen_matrix is None or gen_matrix.size == 0:
        raise ValueError(f"{path}: could not parse mpc.gen")
    if branch_matrix is None or branch_matrix.size == 0:
        raise ValueError(f"{path}: could not parse mpc.branch")
    if gencost_matrix is None or gencost_matrix.size == 0:
        raise ValueError(f"{path}: could not parse mpc.gencost")

    # Pd is column 2 (0-indexed)
    d_ref = bus_matrix[:, BUS_PD_COL].astype(np.float64)
    # Clamp negative demand (some cases have small negative Pd) to 0 for ED
    d_ref = np.maximum(d_ref, 0.0)

    # Only online generators (status == 1), Pmax is column 8
    on = gen_matrix[:, GEN_STATUS_COL] == 1
    p_max = gen_matrix[on, GEN_PMAX_COL].astype(np.float64)
    p_max = np.maximum(p_max, 1e-9)  # avoid zeros for division stability

    # gen_bus: 0-based row index into bus_matrix for each online generator
    # GEN_BUS_COL is bus number (not necessarily 1..n_bus); map to bus_matrix row index
    bus_ids = bus_matrix[:, 0].astype(int)
    bus_id_to_idx = {int(bus_ids[i]): i for i in range(bus_matrix.shape[0])}
    gen_bus_nums = gen_matrix[on, GEN_BUS_COL].astype(int)
    gen_bus = np.array(
        [bus_id_to_idx[int(b)] for b in gen_bus_nums],
        dtype=np.intp,
    )

    branch_matrix = branch_matrix.astype(np.float64)
    gencost_matrix = gencost_matrix.astype(np.float64)
    bus_matrix = bus_matrix.astype(np.float64)
    gen_matrix = gen_matrix.astype(np.float64)

    base_mva = _parse_base_mva(path)

    return (
        d_ref,
        p_max,
        branch_matrix,
        gencost_matrix,
        bus_matrix,
        gen_matrix,
        gen_bus,
        base_mva,
    )


# Thermal violation penalty ($/MW), paper experiments.tex
MTH_DEFAULT = 1500.0


def pglib_to_ref_npz(
    m_path: str | Path,
    out_path: str | Path,
    mth: float = MTH_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert one PGLib .m file to reference case .npz.

    Saves: d_ref, p_max, branch_matrix, gencost_matrix, c_linear, f_min, f_max,
    Phi, PTDF_bus, base_mva, Mth (for SSL loss and thermal violations).

    Returns (d_ref, p_max, branch_matrix, gencost_matrix).
    """
    from opf_utils import (
        build_ptdf_and_phi,
        extract_c_linear,
        extract_f_min_f_max,
    )

    (
        d_ref,
        p_max,
        branch_matrix,
        gencost_matrix,
        bus_matrix,
        gen_matrix,
        gen_bus,
        base_mva,
    ) = parse_pglib_m(m_path)

    on = gen_matrix[:, GEN_STATUS_COL] == 1
    c_linear = extract_c_linear(gencost_matrix, on)
    f_min, f_max = extract_f_min_f_max(branch_matrix, base_mva)
    PTDF_bus, Phi = build_ptdf_and_phi(
        bus_matrix, branch_matrix, gen_bus, base_mva
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        d_ref=d_ref,
        p_max=p_max,
        branch_matrix=branch_matrix,
        gencost_matrix=gencost_matrix,
        c_linear=c_linear,
        f_min=f_min,
        f_max=f_max,
        Phi=Phi,
        PTDF_bus=PTDF_bus,
        base_mva=np.array(base_mva, dtype=np.float64),
        Mth=np.array(mth, dtype=np.float64),
    )
    return d_ref, p_max, branch_matrix, gencost_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create E2ELR datasets from PGLib .m case files in code/data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing PGLib .m files (and where ref/datasets are written).",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Specific case filename (e.g. pglib_opf_case300_ieee.m). If omitted, process all .m in data_dir.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ed", "edr"],
        default="edr",
        help="ED (no reserve) or ED-R (with reserve).",
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=50_000,
        help="Total instances per case (default 50000): 40k train, 5k val, 5k test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if args.case:
        m_files = [data_dir / args.case] if (data_dir / args.case).exists() else [Path(args.case)]
        if not m_files[0].exists():
            raise FileNotFoundError(f"Case file not found: {m_files[0]}")
    else:
        m_files = sorted(data_dir.glob("*.m"))

    if not m_files:
        raise FileNotFoundError(f"No .m files found in {data_dir}")

    train_size = 40_000
    val_size = 5_000
    test_size = 5_000
    if args.n_instances != train_size + val_size + test_size:
        parser.error(
            f"--n_instances must be {train_size + val_size + test_size} (40000+5000+5000)."
        )

    key = jr.PRNGKey(args.seed)

    for m_path in m_files:
        case_stem = m_path.stem  # e.g. pglib_opf_case300_ieee
        ref_path = data_dir / f"{case_stem}_ref.npz"
        out_subdir = data_dir / case_stem

        print(f"Processing {m_path.name} ...")
        try:
            d_ref_np, p_max_np, branch_matrix_np, gencost_matrix_np = pglib_to_ref_npz(m_path, ref_path)
        except Exception as e:
            print(f"  Skip: {e}")
            continue

        d_ref = jnp.array(d_ref_np)
        p_max = jnp.array(p_max_np)
        n_buses, n_gen = d_ref.shape[0], p_max.shape[0]
        n_branches = branch_matrix_np.shape[0] if branch_matrix_np.size else 0
        print(f"  n_buses={n_buses}, n_gen={n_gen}, n_branches={n_branches}, ref saved to {ref_path}")
