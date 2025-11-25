#!/usr/bin/env python3
"""
proton_wavelet_analysis_mf.py

Wavelet analysis of the 3-quark "proton" on the 3D lattice
using the MATRIX-FREE Hamiltonian from proton_matrixfree.py.

This script:
  - Builds the matrix-free 3-quark Hamiltonian (ThreeQuarkHamiltonianMF)
  - Diagonalizes to get the ground-state wavefunction ψ
  - Constructs 3D fields:
      * density_field[x, y, z]  = ⟨n(x,y,z)⟩
      * spin_field[x, y, z]     = ⟨S_z(x,y,z)⟩
  - Performs a 3D discrete wavelet decomposition on each field
  - Prints an energy budget per subband
  - Saves PNGs of:
      * raw density/spin slices
      * mean-subtracted density/spin slices
      * approximation (coarse) wavelet coeffs
      * detail bands at each level
  - Saves a JSON summary of parameters + spectra for external review.

Usage (from substrate_demo folder):

    python proton_wavelet_analysis_mf.py --Lx 3 --Ly 3 --Lz 3 --g_defrag 2.0 --sigma_defrag 0.7
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
import matplotlib.pyplot as plt

try:
    import pywt
except ImportError as e:
    raise SystemExit(
        "PyWavelets (pywt) is required for this script.\n"
        "Install with: pip install pywavelets"
    ) from e

from proton_matrixfree import (
    ThreeQuarkMFParams,
    ThreeQuarkHamiltonianMF,
)
from substrate_engine_3d import site_coords_3d


# =============================================================================
# Ground state + fields
# =============================================================================

def compute_ground_state_matrixfree(
    params: ThreeQuarkMFParams
) -> Tuple[float, np.ndarray, ThreeQuarkHamiltonianMF]:
    """
    Build matrix-free Hamiltonian, run eigsh, return (E0, psi0_flat, Hmf).
    """
    Hmf = ThreeQuarkHamiltonianMF(params)
    dim = Hmf.dim

    print(f"Effective Hilbert dimension dim = {dim}")

    def matvec(v: np.ndarray) -> np.ndarray:
        return Hmf.apply(v)

    H_linop = LinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        dtype=np.complex128,
    )

    print("Running eigsh (matrix-free)...")
    evals, evecs = eigsh(
        H_linop,
        k=params.k_eigs,
        which="SA",
        maxiter=params.max_eigsh_iter,
    )
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    # normalize
    norm = np.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    return E0, psi0, Hmf


def build_density_and_spin_fields(
    params: ThreeQuarkMFParams,
    Hmf: ThreeQuarkHamiltonianMF,
    psi_flat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct 3D fields:
      density_field[x, y, z] = ⟨number of quarks at (x,y,z)⟩
      spin_field[x, y, z]    = ⟨S_z at (x,y,z)⟩
    using the matrix-free ground state ψ.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz

    psi = psi_flat.reshape(Ns, 2, Ns, 2, Ns, 2)
    prob = np.abs(psi) ** 2
    Z = prob.sum()
    if Z <= 0:
        Z = 1.0
    prob /= Z

    density_field = np.zeros((Lx, Ly, Lz), dtype=float)
    spin_field = np.zeros((Lx, Ly, Lz), dtype=float)

    sz_vals = np.array([+0.5, -0.5], dtype=float)

    for r1 in range(Ns):
        x1, y1, z1 = site_coords_3d(r1, Lx, Ly, Lz)
        for r2 in range(Ns):
            x2, y2, z2 = site_coords_3d(r2, Lx, Ly, Lz)
            for r3 in range(Ns):
                x3, y3, z3 = site_coords_3d(r3, Lx, Ly, Lz)

                block = prob[r1, :, r2, :, r3, :]  # (2,2,2)
                p_block = block.sum()
                if p_block < 1e-16:
                    continue

                # add density for each quark
                for (x, y, z) in ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3)):
                    density_field[x, y, z] += p_block

                # spin_z at each site: deposit each quark's S_z
                for s1 in (0, 1):
                    for s2 in (0, 1):
                        for s3 in (0, 1):
                            p_cfg = block[s1, s2, s3]
                            if p_cfg < 1e-16:
                                continue
                            sz1 = sz_vals[s1]
                            sz2 = sz_vals[s2]
                            sz3 = sz_vals[s3]
                            spin_field[x1, y1, z1] += p_cfg * sz1
                            spin_field[x2, y2, z2] += p_cfg * sz2
                            spin_field[x3, y3, z3] += p_cfg * sz3

    # Normalize density so total quarks ≈ 3
    total_n = density_field.sum()
    if total_n > 0:
        density_field *= (3.0 / total_n)

    return density_field, spin_field


# =============================================================================
# Wavelet helpers
# =============================================================================

def wavelet_decompose(
    field: np.ndarray,
    wavelet: str = "db2",
    mode: str = "periodization",
) -> Tuple[Dict[str, Any], list]:
    max_level = pywt.dwtn_max_level(field.shape, wavelet)
    level = max_level if max_level > 0 else 1

    coeffs = pywt.wavedecn(field, wavelet=wavelet, mode=mode, level=level)

    approx = coeffs[0]
    detail_coeffs = coeffs[1:]

    total_energy = np.sum(np.abs(field) ** 2)
    coarse_energy = np.sum(np.abs(approx) ** 2)

    detail_levels = []
    for lev, d in enumerate(detail_coeffs, start=1):
        bands = {}
        level_energy = 0.0
        for band_name, arr in d.items():
            e = float(np.sum(np.abs(arr) ** 2))
            bands[band_name] = e
            level_energy += e
        detail_levels.append({
            "level": lev,
            "bands": bands,
            "total": level_energy,
        })

    spectrum = {
        "total_energy": float(total_energy),
        "coarse_energy": float(coarse_energy),
        "detail_levels": detail_levels,
        "wavelet": wavelet,
        "mode": mode,
    }
    return spectrum, coeffs


def print_wavelet_summary(name: str, spectrum: Dict[str, Any]) -> None:
    print(f"\nWavelet spectrum for {name}:")
    print(f"  wavelet       : {spectrum['wavelet']}")
    print(f"  mode          : {spectrum['mode']}")
    print(f"  total energy  : {spectrum['total_energy']:.6e}")
    frac_coarse = (
        spectrum["coarse_energy"] / spectrum["total_energy"]
        if spectrum["total_energy"] > 0 else 0.0
    )
    print(f"  coarse energy : {spectrum['coarse_energy']:.6e}"
          f"  (fraction = {frac_coarse:.3f})")
    for lvl in spectrum["detail_levels"]:
        lev = lvl["level"]
        total = lvl["total"]
        frac = total / spectrum["total_energy"] if spectrum["total_energy"] > 0 else 0.0
        print(f"  Level {lev}: detail energy = {total:.6e} (fraction = {frac:.3f})")
        for band, e in sorted(lvl["bands"].items()):
            print(f"    band {band:3s} : {e:.6e}")


def _middle_slice(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        z = arr.shape[2] // 2
        return arr[:, :, z]
    elif arr.ndim == 2:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1)


def save_base_field_pngs(
    density_field: np.ndarray,
    spin_field: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Save PNGs of raw density/spin and mean-subtracted versions.
    """

    # ---- density ----
    dens_slice = _middle_slice(density_field)
    dens_mean = dens_slice.mean()
    dens_dev = dens_slice - dens_mean

    fig = plt.figure()
    plt.imshow(dens_slice, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.title("Proton density (mid-z slice, absolute)")
    png_path = out_dir / "proton_mf_density_slice_abs.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved density (abs) slice PNG -> {png_path}")

    fig = plt.figure()
    max_abs = np.max(np.abs(dens_dev))
    if max_abs > 0:
        plt.imshow(dens_dev, origin="lower", cmap="seismic",
                   vmin=-max_abs, vmax=max_abs)
    else:
        plt.imshow(dens_dev, origin="lower", cmap="seismic")
    plt.colorbar()
    plt.title("Proton density deviation (mid-z slice, mean-subtracted)")
    png_path = out_dir / "proton_mf_density_slice_dev.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved density (dev) slice PNG -> {png_path}")

    # ---- spin_z ----
    spin_slice = _middle_slice(spin_field)
    spin_mean = spin_slice.mean()
    spin_dev = spin_slice - spin_mean

    fig = plt.figure()
    max_abs = np.max(np.abs(spin_dev))
    if max_abs > 0:
        plt.imshow(spin_dev, origin="lower", cmap="seismic",
                   vmin=-max_abs, vmax=max_abs)
    else:
        plt.imshow(spin_dev, origin="lower", cmap="seismic")
    plt.colorbar()
    plt.title("Proton S_z deviation (mid-z slice, mean-subtracted)")
    png_path = out_dir / "proton_mf_spin_z_slice_dev.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spin_z (dev) slice PNG -> {png_path}")


def save_coeff_slices_as_pngs(
    field_name: str,
    coeffs: list,
    out_dir: Path,
) -> None:
    """
    Save PNGs of:
      - approximation coeffs (coarse, mean-subtracted)
      - each detail band (signed, diverging colormap).
    """
    approx = coeffs[0]
    detail_coeffs = coeffs[1:]

    # Approximation
    approx_slice = _middle_slice(approx)
    approx_centered = approx_slice - approx_slice.mean()
    fig = plt.figure()
    max_abs = np.max(np.abs(approx_centered))
    if max_abs > 0:
        plt.imshow(approx_centered, origin="lower", cmap="seismic",
                   vmin=-max_abs, vmax=max_abs)
    else:
        plt.imshow(approx_centered, origin="lower", cmap="seismic")
    plt.colorbar()
    plt.title(f"{field_name} - approximation (coarse, mean-subtracted)")
    png_path = out_dir / f"proton_mf_wavelet_{field_name}_approx.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {field_name} approximation PNG -> {png_path}")

    # Details
    for level_idx, d in enumerate(detail_coeffs, start=1):
        for band_name, arr in d.items():
            arr_slice = _middle_slice(arr)
            fig = plt.figure()
            max_abs = np.max(np.abs(arr_slice))
            if max_abs > 0:
                plt.imshow(arr_slice, origin="lower", cmap="seismic",
                           vmin=-max_abs, vmax=max_abs)
            else:
                plt.imshow(arr_slice, origin="lower", cmap="seismic")
            plt.colorbar()
            plt.title(f"{field_name} - level {level_idx}, band {band_name}")
            png_path = out_dir / f"proton_mf_wavelet_{field_name}_L{level_idx}_{band_name}.png"
            fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {field_name} L{level_idx} band {band_name} PNG -> {png_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Wavelet decomposition of the matrix-free toy 3-quark proton density and spin fields."
    )
    parser.add_argument("--Lx", type=int, default=3)
    parser.add_argument("--Ly", type=int, default=3)
    parser.add_argument("--Lz", type=int, default=3)
    parser.add_argument("--J_hop", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--g_defrag", type=float, default=2.0)
    parser.add_argument("--sigma_defrag", type=float, default=0.7)
    parser.add_argument("--lambda_G", type=float, default=5.0)
    parser.add_argument("--B", type=float, default=0.0)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wavelet", type=str, default="db2")
    parser.add_argument("--mode", type=str, default="periodization")
    args = parser.parse_args()

    params = ThreeQuarkMFParams(
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        J_hop=args.J_hop,
        m=args.m,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        lambda_G=args.lambda_G,
        B_field=args.B,
        max_eigsh_iter=args.max_iter,
        seed=args.seed,
    )

    np.random.seed(params.seed)

    print("\n" + "=" * 70)
    print("MATRIX-FREE 3-QUARK PROTON WAVELET ANALYSIS")
    print("=" * 70)
    print("Parameters:")
    for k, v in vars(params).items():
        print(f"  {k:12s} = {v}")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ground state
    E0, psi0, Hmf = compute_ground_state_matrixfree(params)
    print(f"\nGround state energy E0 = {E0:.6f}")

    # fields
    density_field, spin_field = build_density_and_spin_fields(params, Hmf, psi0)
    print(f"Density field shape     = {density_field.shape}")
    print(f"Spin field shape        = {spin_field.shape}")

    # raw field PNGs
    print("\nSaving base field PNGs...")
    save_base_field_pngs(density_field, spin_field, out_dir)

    # wavelet decomposition
    spec_density, coeffs_density = wavelet_decompose(
        density_field,
        wavelet=args.wavelet,
        mode=args.mode,
    )
    spec_spin, coeffs_spin = wavelet_decompose(
        spin_field,
        wavelet=args.wavelet,
        mode=args.mode,
    )

    print_wavelet_summary("density", spec_density)
    print_wavelet_summary("spin_z", spec_spin)

    print("\nSaving wavelet coefficient PNGs for density...")
    save_coeff_slices_as_pngs("density", coeffs_density, out_dir)

    print("\nSaving wavelet coefficient PNGs for spin_z...")
    save_coeff_slices_as_pngs("spin_z", coeffs_spin, out_dir)

    # save raw fields
    np.savez(
        str(out_dir / "proton_mf_wavelet_fields.npz"),
        density_field=density_field,
        spin_field=spin_field,
    )
    print(f"\nSaved raw fields to: {out_dir / 'proton_mf_wavelet_fields.npz'}")

    # save a JSON summary so you (or I) can review numerically
    summary = {
        "params": {
            "Lx": params.Lx,
            "Ly": params.Ly,
            "Lz": params.Lz,
            "J_hop": params.J_hop,
            "m": params.m,
            "g_defrag": params.g_defrag,
            "sigma_defrag": params.sigma_defrag,
            "lambda_G": params.lambda_G,
            "B_field": params.B_field,
            "max_eigsh_iter": params.max_eigsh_iter,
            "seed": params.seed,
            "wavelet": args.wavelet,
            "mode": args.mode,
        },
        "ground_state": {
            "E0": float(E0),
        },
        "density_stats": {
            "min": float(density_field.min()),
            "max": float(density_field.max()),
            "mean": float(density_field.mean()),
            "std": float(density_field.std()),
        },
        "spin_stats": {
            "min": float(spin_field.min()),
            "max": float(spin_field.max()),
            "mean": float(spin_field.mean()),
            "std": float(spin_field.std()),
        },
        "density_wavelet_spectrum": spec_density,
        "spin_wavelet_spectrum": spec_spin,
    }

    summary_path = out_dir / "proton_mf_wavelet_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to: {summary_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
