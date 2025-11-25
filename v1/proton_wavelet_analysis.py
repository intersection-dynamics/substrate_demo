#!/usr/bin/env python3
"""
proton_wavelet_analysis.py

Wavelet analysis of the 3-quark "proton" on the 3D substrate.

This script:
  - Builds the 3-quark Hamiltonian (via proton_simulator.py)
  - Diagonalizes to get the ground-state wavefunction ψ
  - Constructs:
      * density_field[x, y, z]  = ⟨n(x,y,z)⟩
      * spin_field[x, y, z]     = ⟨S_z(x,y,z)⟩
  - Performs a 3D discrete wavelet decomposition on each field
  - Prints an energy budget per subband
  - Saves PNGs of:
      * raw density and spin fields (mid-z slice)
      * mean-subtracted density / spin (mid-z slice)
      * approximation (coarse) coefficients (mean-subtracted)
      * each detail band at each level (signed, diverging colormap)
  - Saves raw fields in outputs/ for further analysis

Usage (from substrate_demo folder):

    python proton_wavelet_analysis.py
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

try:
    import pywt
except ImportError as e:
    raise SystemExit(
        "PyWavelets (pywt) is required for this script.\n"
        "Install with: pip install pywavelets"
    ) from e

from proton_simulator import (
    ThreeQuarkParams,
    build_threequark_hamiltonian,
    decode_basis_3q,
)
from substrate_engine_3d import site_coords_3d


# =============================================================================
# Ground-state + field construction
# =============================================================================

def compute_ground_state_wavefunction(params: ThreeQuarkParams) -> Tuple[float, np.ndarray]:
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = (Ns * 2) ** 3

    print(f"Building 3-quark Hamiltonian: dim = {dim}")
    H = build_threequark_hamiltonian(params)

    print("Diagonalizing ground state with eigsh...")
    evals, evecs = eigsh(H, k=params.k_eigs, which="SA", maxiter=params.max_eigsh_iter)
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    # Normalize explicitly
    norm = np.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    return E0, psi0


def build_density_and_spin_fields(
    params: ThreeQuarkParams,
    psi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    density_field[x, y, z] = ⟨number of quarks at (x,y,z)⟩
    spin_field[x, y, z]    = ⟨S_z at (x,y,z)⟩
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = psi.shape[0]

    density_field = np.zeros((Lx, Ly, Lz), dtype=float)
    spin_field = np.zeros((Lx, Ly, Lz), dtype=float)

    for idx in range(dim):
        amp = psi[idx]
        p = float(np.abs(amp) ** 2)
        if p < 1e-14:
            continue

        r1, s1, r2, s2, r3, s3 = decode_basis_3q(idx, Ns)

        r_list = [r1, r2, r3]
        s_list = [s1, s2, s3]

        for r, s in zip(r_list, s_list):
            x, y, z = site_coords_3d(r, Lx, Ly, Lz)
            density_field[x, y, z] += p
            sz = +0.5 if s == 0 else -0.5
            spin_field[x, y, z] += p * sz

    # Normalize density so total quarks ≈ 3
    total_n = density_field.sum()
    if total_n > 0:
        density_field *= (3.0 / total_n)

    return density_field, spin_field


# =============================================================================
# Wavelet analysis helpers
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
    Save PNGs of raw density/spin and their mean-subtracted versions.
    """
    # ---------- Density ----------
    dens_slice = _middle_slice(density_field)
    dens_mean = dens_slice.mean()
    dens_dev = dens_slice - dens_mean

    # raw
    fig = plt.figure()
    plt.imshow(dens_slice, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.title("Proton density (mid-z slice, absolute)")
    png_path = out_dir / "proton_density_slice_abs.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved density (abs) slice PNG -> {png_path}")

    # mean-subtracted
    fig = plt.figure()
    max_abs = np.max(np.abs(dens_dev))
    if max_abs > 0:
        plt.imshow(dens_dev, origin="lower", cmap="seismic",
                   vmin=-max_abs, vmax=max_abs)
    else:
        plt.imshow(dens_dev, origin="lower", cmap="seismic")
    plt.colorbar()
    plt.title("Proton density deviation (mid-z slice, mean-subtracted)")
    png_path = out_dir / "proton_density_slice_dev.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved density (dev) slice PNG -> {png_path}")

    # ---------- Spin_z ----------
    spin_slice = _middle_slice(spin_field)
    spin_mean = spin_slice.mean()
    spin_dev = spin_slice - spin_mean

    # mean-subtracted only (absolute values are tiny)
    fig = plt.figure()
    max_abs = np.max(np.abs(spin_dev))
    if max_abs > 0:
        plt.imshow(spin_dev, origin="lower", cmap="seismic",
                   vmin=-max_abs, vmax=max_abs)
    else:
        plt.imshow(spin_dev, origin="lower", cmap="seismic")
    plt.colorbar()
    plt.title("Proton S_z deviation (mid-z slice, mean-subtracted)")
    png_path = out_dir / "proton_spin_z_slice_dev.png"
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

    # Approximation (coarse): mean-subtracted
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
    png_path = out_dir / f"proton_wavelet_{field_name}_approx.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {field_name} approximation PNG -> {png_path}")

    # Details per level/band
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
            png_path = out_dir / f"proton_wavelet_{field_name}_L{level_idx}_{band_name}.png"
            fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {field_name} L{level_idx} band {band_name} PNG -> {png_path}")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Wavelet decomposition of the toy 3-quark proton density and spin fields."
    )
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=2)
    parser.add_argument("--Lz", type=int, default=2)
    parser.add_argument("--J_hop", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--g_defrag", type=float, default=1.0)
    parser.add_argument("--sigma_defrag", type=float, default=1.0)
    parser.add_argument("--lambda_G", type=float, default=5.0)
    parser.add_argument("--lambda_S", type=float, default=-1.0)
    parser.add_argument("--lambda_T", type=float, default=0.0)
    parser.add_argument("--J_exch", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=0.0)
    parser.add_argument("--max_iter", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wavelet", type=str, default="db2")
    parser.add_argument("--mode", type=str, default="periodization")
    args = parser.parse_args()

    np.random.seed(args.seed)

    params = ThreeQuarkParams(
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        J_hop=args.J_hop,
        m=args.m,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        lambda_G=args.lambda_G,
        lambda_S=args.lambda_S,
        lambda_T=args.lambda_T,
        J_exch=args.J_exch,
        B_field=args.B,
        max_eigsh_iter=args.max_iter,
        k_eigs=1,
    )

    print("\n" + "=" * 70)
    print("3-QUARK PROTON WAVELET ANALYSIS")
    print("=" * 70)
    print("Parameters:")
    for k, v in vars(params).items():
        print(f"  {k:12s} = {v}")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute ground state and fields
    E0, psi0 = compute_ground_state_wavefunction(params)
    density_field, spin_field = build_density_and_spin_fields(params, psi0)

    print(f"\nGround state energy E0 = {E0:.6f}")
    print(f"Density field shape     = {density_field.shape}")
    print(f"Spin field shape        = {spin_field.shape}")

    # Save base field PNGs
    print("\nSaving raw field PNGs...")
    save_base_field_pngs(density_field, spin_field, out_dir)

    # Wavelet spectra + coeffs
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

    # Save PNGs for density and spin wavelet coeffs
    print("\nSaving wavelet coefficient PNGs for density...")
    save_coeff_slices_as_pngs("density", coeffs_density, out_dir)

    print("\nSaving wavelet coefficient PNGs for spin_z...")
    save_coeff_slices_as_pngs("spin_z", coeffs_spin, out_dir)

    # Save raw fields for later analysis
    np.savez(
        str(out_dir / "proton_wavelet_fields.npz"),
        density_field=density_field,
        spin_field=spin_field,
    )
    print(f"\nSaved raw fields to: {out_dir / 'proton_wavelet_fields.npz'}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
