#!/usr/bin/env python3
"""
spinor_finite_size_scan.py

Finite-size scaling harness for the spinor noise→emergent model.

It reuses the GPU evolution code from substrate_spinor_noise_emergent_gpu.py
to run the same physics at multiple cubic lattice sizes Lx=Ly=Lz=L and
collect:

  - final r_eff
  - r_eff / L
  - radial density profile n(r)

Results are printed as a table and saved to a small .npz file for later plotting.
"""

import os
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np

# We reuse the evolution + diagnostics from the main spinor script.
# Make sure this file lives in the same directory as substrate_spinor_noise_emergent_gpu.py
from substrate_spinor_noise_emergent_gpu import (
    evolve_spinor_from_noise_gpu,
    compute_r_eff,
    compute_radial_profile,
    ensure_outdir,
)


@dataclass
class SpinorFSParams:
    """
    Parameters for a single finite-size run.

    These mirror the arguments of evolve_spinor_from_noise_gpu, except Nx,Ny,Nz
    which we set per L.
    """
    dt: float = 0.01
    n_steps: int = 5000
    m_eff: float = 1.0
    alpha: float = 0.2
    beta: float = 1.0
    g_defrag: float = -4.0
    sigma_defrag: float = 3.0
    omega_conf: float = 0.05
    gamma_spin: float = 0.3
    snapshot_every: int = 0  # 0 → no XY snapshots (faster for scans)


def run_single_L(
    L: int,
    base_params: SpinorFSParams,
    outdir: str,
) -> Dict[str, Any]:
    """
    Run the spinor model at Lx=Ly=Lz=L and return summary diagnostics.
    """
    print("=" * 70)
    print(f"FINITE-SIZE RUN: L = {L}")
    print("=" * 70)

    # Unpack base params
    p = base_params

    # Core evolution: no snapshots to keep it light.
    density_tot, m_z, r_eff_history = evolve_spinor_from_noise_gpu(
        Nx=L,
        Ny=L,
        Nz=L,
        dt=p.dt,
        n_steps=p.n_steps,
        m_eff=p.m_eff,
        alpha=p.alpha,
        beta=p.beta,
        g_defrag=p.g_defrag,
        sigma_defrag=p.sigma_defrag,
        omega_conf=p.omega_conf,
        gamma_spin=p.gamma_spin,
        snapshot_every=p.snapshot_every,
        outdir=outdir,
    )

    # Final diagnostics on CPU
    final_r_eff = compute_r_eff(density_tot)
    r_over_L = final_r_eff / float(L)

    r_centers, n_of_r = compute_radial_profile(density_tot, n_bins=40)

    print(f"[L={L}] Final r_eff       ≈ {final_r_eff:.4f}")
    print(f"[L={L}] r_eff / L        ≈ {r_over_L:.4f}")
    print(f"[L={L}] # of checkpoints = {len(r_eff_history)}")
    if len(r_eff_history) > 0:
        print(f"[L={L}] r_eff (first checkpoint) ≈ {r_eff_history[0]:.4f}")
        print(f"[L={L}] r_eff (last  checkpoint) ≈ {r_eff_history[-1]:.4f}")

    # Package results
    result = {
        "L": L,
        "final_r_eff": float(final_r_eff),
        "r_eff_over_L": float(r_over_L),
        "r_eff_history": np.array(r_eff_history, dtype=float),
        "r_centers": np.array(r_centers, dtype=float),
        "n_of_r": np.array(n_of_r, dtype=float),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Finite-size scaling harness for substrate_spinor_noise_emergent_gpu.\n"
            "Runs the same couplings at multiple cubic lattice sizes Lx=Ly=Lz=L."
        )
    )

    # Which sizes to scan
    parser.add_argument(
        "--L_list",
        type=int,
        nargs="+",
        default=[64, 96, 128],
        help="List of cubic lattice sizes L to scan (default: 64 96 128)",
    )

    # Imaginary-time / physics parameters (same meaning as in substrate_spinor_noise_emergent_gpu.py)
    parser.add_argument("--dt", type=float, default=0.01, help="Imaginary-time step")
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--m_eff", type=float, default=1.0, help="Effective mass")

    parser.add_argument("--alpha", type=float, default=0.2, help="Coefficient of |psi|^2")
    parser.add_argument("--beta", type=float, default=1.0, help="Coefficient of |psi|^4")

    parser.add_argument(
        "--g_defrag",
        type=float,
        default=-4.0,
        help="Strength of nonlocal defrag term (negative = attraction)",
    )
    parser.add_argument(
        "--sigma_defrag",
        type=float,
        default=3.0,
        help="Width of Gaussian defrag kernel",
    )

    parser.add_argument(
        "--omega_conf",
        type=float,
        default=0.05,
        help="Harmonic confining strength (0 to disable)",
    )
    parser.add_argument(
        "--gamma_spin",
        type=float,
        default=0.3,
        help="Spin coupling strength gamma_spin",
    )

    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=0,
        help="Snapshots every N steps (0 = never; default 0 for fast scans)",
    )

    args = parser.parse_args()

    # Build shared parameter struct
    base_params = SpinorFSParams(
        dt=args.dt,
        n_steps=args.n_steps,
        m_eff=args.m_eff,
        alpha=args.alpha,
        beta=args.beta,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        omega_conf=args.omega_conf,
        gamma_spin=args.gamma_spin,
        snapshot_every=args.snapshot_every,
    )

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = ensure_outdir(script_dir, "outputs_finite_size")

    print("Finite-size scan parameters (shared across L):")
    for k, v in asdict(base_params).items():
        print(f"  {k:15s} = {v}")
    print(f"Results will be saved under: {outdir}")
    print("-" * 70)

    all_results: List[Dict[str, Any]] = []

    for L in args.L_list:
        res_L = run_single_L(L, base_params, outdir)
        all_results.append(res_L)

    # Aggregate into arrays for saving
    L_array = np.array([r["L"] for r in all_results], dtype=int)
    r_eff_array = np.array([r["final_r_eff"] for r in all_results], dtype=float)
    r_over_L_array = np.array([r["r_eff_over_L"] for r in all_results], dtype=float)

    # Print a clean summary table
    print("\n" + "=" * 70)
    print("FINITE-SIZE SUMMARY (spinor noise → emergent blob)")
    print("=" * 70)
    print("  L        r_eff        r_eff/L")
    print("  ---------------------------------")
    for L, reff, reffL in zip(L_array, r_eff_array, r_over_L_array):
        print(f"  {L:3d}    {reff:8.4f}    {reffL:8.4f}")
    print("  ---------------------------------")

    # Save to npz for later plotting / paper figures
    save_path = os.path.join(outdir, "spinor_finite_size_results.npz")
    np.savez_compressed(
        save_path,
        L=L_array,
        r_eff=r_eff_array,
        r_eff_over_L=r_over_L_array,
        # store per-L histories and radial profiles as an object array
        results=np.array(all_results, dtype=object),
    )
    print(f"\nSaved finite-size results to: {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
