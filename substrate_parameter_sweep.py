#!/usr/bin/env python3
"""
substrate_parameter_sweep.py

Parameter sweep driver for the finite-Hilbert substrate in
ur_substrate_quantum_lab_v2.py.

Sweeps over (lambda_G, lambda_S, J_exch) on a small lattice (e.g. 2x2, 3x3),
calls the two-excitation ground-state solver, and writes a CSV with:

  - Lx, Ly
  - J_hop, m, g_defrag, sigma_defrag
  - lambda_G, lambda_S, lambda_T, J_exch
  - E0 (ground state energy)
  - antisym_score, sym_score
  - overlap_prob
  - singlet_fraction
  - E_gauss
  - converged (True/False)
  - error_msg (if any)

Usage example (from repo root):

    python substrate_parameter_sweep.py ^
        --Lx 2 --Ly 2 ^
        --lambda-G-min 0.0 --lambda-G-max 6.0 --n-lambda-G 4 ^
        --lambda-S-min -2.0 --lambda-S-max 0.0 --n-lambda-S 3 ^
        --J-exch-min 0.0 --J-exch-max 2.0 --n-J-exch 3 ^
        --output sweep_2x2.csv

This script assumes ur_substrate_quantum_lab_v2.py is in the same directory
and provides SubstrateParams and run_substrate_ground_state().
"""

import argparse
import csv
import math
import sys
import traceback
from dataclasses import asdict
from typing import List, Tuple

import numpy as np

# Import from the ur-script
try:
    from ur_substrate_quantum_lab_v2 import SubstrateParams, run_substrate_ground_state
except ImportError as e:
    print("ERROR: Could not import ur_substrate_quantum_lab_v2. "
          "Make sure this script is in the same directory as ur_substrate_quantum_lab_v2.py.")
    print("ImportError:", e)
    sys.exit(1)


def linspace_inclusive(vmin: float, vmax: float, n: int) -> List[float]:
    """
    Inclusive linspace: returns n points from vmin to vmax inclusive.
    If n == 1, returns [vmin].
    """
    if n <= 1:
        return [vmin]
    return list(np.linspace(vmin, vmax, n))


def parse_args():
    p = argparse.ArgumentParser(
        description="Parameter sweep for finite-Hilbert substrate "
                    "using ur_substrate_quantum_lab_v2.py"
    )

    # Lattice size
    p.add_argument("--Lx", type=int, default=2, help="Lattice size in x.")
    p.add_argument("--Ly", type=int, default=2, help="Lattice size in y.")

    # Base substrate params (not swept)
    p.add_argument("--J-hop", type=float, default=1.0, dest="J_hop",
                   help="Hopping strength J_hop.")
    p.add_argument("--m", type=float, default=0.1,
                   help="Mass term per excitation.")
    p.add_argument("--g-defrag", type=float, default=1.0,
                   help="Defrag strength g_defrag.")
    p.add_argument("--sigma-defrag", type=float, default=1.0,
                   help="Defrag Gaussian width.")
    p.add_argument("--lambda-T", type=float, default=0.0, dest="lambda_T",
                   help="Triplet penalty at overlap.")

    # Sweep over lambda_G, lambda_S, J_exch
    p.add_argument("--lambda-G-min", type=float, default=0.0, dest="lambda_G_min",
                   help="Minimum lambda_G.")
    p.add_argument("--lambda-G-max", type=float, default=6.0, dest="lambda_G_max",
                   help="Maximum lambda_G.")
    p.add_argument("--n-lambda-G", type=int, default=4,
                   help="Number of lambda_G sample points.")

    p.add_argument("--lambda-S-min", type=float, default=-2.0, dest="lambda_S_min",
                   help="Minimum lambda_S.")
    p.add_argument("--lambda-S-max", type=float, default=0.0, dest="lambda_S_max",
                   help="Maximum lambda_S.")
    p.add_argument("--n-lambda-S", type=int, default=3,
                   help="Number of lambda_S sample points.")

    p.add_argument("--J-exch-min", type=float, default=0.0, dest="J_exch_min",
                   help="Minimum J_exch.")
    p.add_argument("--J-exch-max", type=float, default=2.0, dest="J_exch_max",
                   help="Maximum J_exch.")
    p.add_argument("--n-J-exch", type=int, default=3,
                   help="Number of J_exch sample points.")

    # Eigsh iterations (pass-through)
    p.add_argument("--max-eigsh-iter", type=int, default=5000,
                   help="Maximum iterations for eigsh.")

    # Output CSV
    p.add_argument("--output", type=str, default="substrate_sweep.csv",
                   help="Output CSV filename.")

    args = p.parse_args()
    return args


def main():
    args = parse_args()

    # Build ranges
    lambda_G_vals = linspace_inclusive(args.lambda_G_min, args.lambda_G_max,
                                       args.n_lambda_G)
    lambda_S_vals = linspace_inclusive(args.lambda_S_min, args.lambda_S_max,
                                       args.n_lambda_S)
    J_exch_vals = linspace_inclusive(args.J_exch_min, args.J_exch_max,
                                     args.n_J_exch)

    total_points = len(lambda_G_vals) * len(lambda_S_vals) * len(J_exch_vals)

    print("======================================================================")
    print("SUBSTRATE PARAMETER SWEEP")
    print("======================================================================")
    print(f"Lattice: Lx={args.Lx}, Ly={args.Ly}")
    print(f"Total points: {total_points}")
    print("lambda_G range:", lambda_G_vals)
    print("lambda_S range:", lambda_S_vals)
    print("J_exch range  :", J_exch_vals)
    print("Output CSV    :", args.output)
    print("======================================================================")

    # CSV header
    header = [
        "Lx", "Ly",
        "J_hop", "m", "g_defrag", "sigma_defrag",
        "lambda_G", "lambda_S", "lambda_T", "J_exch",
        "E0",
        "antisym_score", "sym_score",
        "overlap_prob", "singlet_fraction",
        "E_gauss",
        "converged", "error_msg",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        point_index = 0

        for lambda_G in lambda_G_vals:
            for lambda_S in lambda_S_vals:
                for J_exch in J_exch_vals:
                    point_index += 1
                    print(f"[{point_index:4d}/{total_points:4d}] "
                          f"lambda_G={lambda_G:6.3f}, "
                          f"lambda_S={lambda_S:6.3f}, "
                          f"J_exch={J_exch:6.3f}")

                    # Build substrate params for this point
                    sub_params = SubstrateParams(
                        Lx=args.Lx,
                        Ly=args.Ly,
                        J_hop=args.J_hop,
                        m=args.m,
                        g_defrag=args.g_defrag,
                        sigma_defrag=args.sigma_defrag,
                        lambda_G=lambda_G,
                        lambda_S=lambda_S,
                        lambda_T=args.lambda_T,
                        J_exch=J_exch,
                        max_eigsh_iter=args.max_eigsh_iter,
                    )

                    # Try solving; catch failures
                    converged = True
                    error_msg = ""
                    E0 = float("nan")
                    antisym_score = float("nan")
                    sym_score = float("nan")
                    overlap_prob = float("nan")
                    singlet_fraction = float("nan")
                    E_gauss = float("nan")

                    try:
                        res = run_substrate_ground_state(sub_params)
                        E0 = res.get("E0", float("nan"))
                        antisym_score = res.get("antisym_score", float("nan"))
                        sym_score = res.get("sym_score", float("nan"))
                        overlap_prob = res.get("overlap_prob", float("nan"))
                        singlet_fraction = res.get("singlet_fraction", float("nan"))
                        E_gauss = res.get("E_gauss", float("nan"))
                    except Exception as e:
                        converged = False
                        error_msg = f"{type(e).__name__}: {e}"
                        traceback.print_exc()

                    row = [
                        args.Lx, args.Ly,
                        args.J_hop, args.m, args.g_defrag, args.sigma_defrag,
                        lambda_G, lambda_S, args.lambda_T, J_exch,
                        E0,
                        antisym_score, sym_score,
                        overlap_prob, singlet_fraction,
                        E_gauss,
                        converged, error_msg,
                    ]
                    writer.writerow(row)

    print("======================================================================")
    print("Sweep complete.")
    print(f"CSV written to: {args.output}")
    print("======================================================================")


if __name__ == "__main__":
    main()
