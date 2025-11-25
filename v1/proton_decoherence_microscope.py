#!/usr/bin/env python3
"""
proton_decoherence_microscope.py

"Decoherence microscope" for the 3-quark proton toy:

  1. Runs proton_matrixfree.py to obtain the 3D density n(x,y,z).
  2. Performs a 3D wavelet transform of the density.
  3. Assigns decoherence rates γ_band that grow with spatial frequency
     (number of 'd' characters in the band key).
  4. Simulates band-wise exponential decay:
        c_band(t) = c_band(0) * exp(-γ_band t)
     and reconstructs n(x,y,z; t) via inverse DWT.
  5. Saves:
        - XY mid-plane slices at multiple t as PNGs
        - radial density profiles at multiple t
        - a bar chart showing γ_band per wavelet band

This is a toy decoherence model on the *density field*, not a full
Lindblad evolution of the quantum state, but it gives an intuitive
"microscope" on which internal spatial structures are most fragile.

Example (Windows):

    python proton_decoherence_microscope.py ^
      --Lx 4 --Ly 4 --Lz 4 ^
      --g_defrag 2.0 --sigma_defrag 0.7 ^
      --lambda_G 5.0 --max_iter 2000 --seed 42 ^
      --t_max 2.0 --n_times 6
"""

import argparse
import os
import re
import subprocess
import math
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt

# Wavelet support (required here)
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


# ---------------------------------------------------------------------
# Regex for parsing proton_matrixfree.py output
# ---------------------------------------------------------------------

OCCUPANCY_LINE_RE = re.compile(
    r"r=\s*(\d+)\s+\(x,y,z\)=\((\d+),(\d+),(\d+)\)\s*:\s*n\(r\)\s*≈\s*([0-9.]+)"
)

BARYCENTER_LINE_RE = re.compile(
    r"Barycenter \(x,y,z\): \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)"
)


# ---------------------------------------------------------------------
# Running underlying proton_matrixfree.py
# ---------------------------------------------------------------------

def run_proton_matrixfree(
    script_path: str,
    args_ns: argparse.Namespace
) -> str:
    """
    Call proton_matrixfree.py as a subprocess, capture stdout, and return it.

    Also prints the stdout back to this process's stdout so the user sees
    the normal diagnostics.

    Forces PYTHONIOENCODING=utf-8 to avoid Unicode issues with '≈' on Windows.
    """
    cmd = ["python", script_path]

    def add_flag(flag: str, value):
        if value is None:
            return
        cmd.extend([flag, str(value)])

    add_flag("--Lx", args_ns.Lx)
    add_flag("--Ly", args_ns.Ly)
    add_flag("--Lz", args_ns.Lz)
    add_flag("--J_hop", args_ns.J_hop)
    add_flag("--m", args_ns.m)
    add_flag("--g_defrag", args_ns.g_defrag)
    add_flag("--sigma_defrag", args_ns.sigma_defrag)
    add_flag("--lambda_G", args_ns.lambda_G)
    add_flag("--B", args_ns.B)
    add_flag("--max_iter", args_ns.max_iter)
    add_flag("--seed", args_ns.seed)

    print("=" * 70)
    print("RUNNING proton_matrixfree.py")
    print("Command:", " ".join(cmd))
    print("=" * 70)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print("\n[stderr from proton_matrixfree.py]", completed.stderr, sep="\n")

    if completed.returncode != 0:
        raise RuntimeError(
            f"proton_matrixfree.py exited with code {completed.returncode}"
        )

    return completed.stdout


def parse_proton_output(
    text: str,
    Lx: int,
    Ly: int,
    Lz: int
) -> Dict[str, Any]:
    """
    Parse proton_matrixfree.py output to extract:

      - density[x,y,z]
      - barycenter (if present)

    Returns dict with keys:
      "density": np.ndarray of shape (Lx, Ly, Lz)
      "barycenter": (bx, by, bz) or None
    """
    density = np.zeros((Lx, Ly, Lz), dtype=float)
    barycenter = None

    # Barycenter (if printed)
    for line in text.splitlines():
        m = BARYCENTER_LINE_RE.search(line)
        if m:
            bx, by, bz = map(float, m.groups())
            barycenter = (bx, by, bz)
            break

    # Site occupancy
    count = 0
    for line in text.splitlines():
        m = OCCUPANCY_LINE_RE.search(line)
        if not m:
            continue
        _, xs, ys, zs, val = m.groups()
        x = int(xs)
        y = int(ys)
        z = int(zs)
        n_r = float(val)
        if x < 0 or x >= Lx or y < 0 or y >= Ly or z < 0 or z >= Lz:
            raise ValueError(
                f"Occupancy (x,y,z)=({x},{y},{z}) out of bounds for "
                f"L=({Lx},{Ly},{Lz})"
            )
        density[x, y, z] = n_r
        count += 1

    if count == 0:
        raise ValueError("No site occupancy lines parsed from output.")

    total = density.sum()
    print(f"\n[decoherence_microscope] Parsed {count} sites, total density = {total:.6f}")
    return {"density": density, "barycenter": barycenter}


# ---------------------------------------------------------------------
# Helpers: outdir, slices, radial profile
# ---------------------------------------------------------------------

def ensure_outdir(base_dir: str, subdir: str = "outputs") -> str:
    out_path = os.path.join(base_dir, subdir)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def compute_radial_profile(
    density: np.ndarray,
    barycenter: Optional[Tuple[float, float, float]] = None,
    n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    Lx, Ly, Lz = density.shape
    xs = np.arange(Lx)
    ys = np.arange(Ly)
    zs = np.arange(Lz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    if barycenter is None:
        bx = 0.5 * (Lx - 1)
        by = 0.5 * (Ly - 1)
        bz = 0.5 * (Lz - 1)
    else:
        bx, by, bz = barycenter

    R = np.sqrt((X - bx) ** 2 + (Y - by) ** 2 + (Z - bz) ** 2)
    r_max = R.max()
    bins = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    n_of_r = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)

    flat_R = R.flatten()
    flat_n = density.flatten()
    inds = np.digitize(flat_R, bins) - 1

    for rv, nv, idx in zip(flat_R, flat_n, inds):
        if 0 <= idx < n_bins:
            n_of_r[idx] += nv
            counts[idx] += 1.0

    mask = counts > 0
    n_of_r[mask] /= counts[mask]
    return r_centers, n_of_r


def plot_xy_slice(
    density: np.ndarray,
    t_label: str,
    outdir: str,
    prefix: str = "proton_decoherence"
) -> str:
    """
    Save an XY slice at mid-z for a given density field.
    """
    Lx, Ly, Lz = density.shape
    cz = Lz // 2
    slice_xy = density[:, :, cz]

    plt.figure(figsize=(4, 4))
    plt.imshow(slice_xy.T, origin="lower", interpolation="nearest")
    plt.colorbar(label="n(x,y,z_mid)")
    plt.title(f"XY slice at mid-z, t={t_label}")
    plt.xlabel("x")
    plt.ylabel("y")
    fname = f"{prefix}_slice_xy_t{t_label.replace('.', 'p')}.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_radial_profiles_vs_time(
    r_centers: np.ndarray,
    profiles: List[np.ndarray],
    t_values: List[float],
    outdir: str,
    prefix: str = "proton_decoherence"
):
    plt.figure(figsize=(5, 4))
    for n_of_r, t in zip(profiles, t_values):
        plt.plot(r_centers, n_of_r, marker="o", label=f"t={t:.2f}")
    plt.xlabel("Radius r (lattice units)")
    plt.ylabel("Average density n(r)")
    plt.title("Radial density vs decoherence time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(outdir, f"{prefix}_radial_profiles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[decoherence_microscope] Saved radial profiles to {path}")


def plot_gamma_bands(
    band_names: List[str],
    gammas: List[float],
    outdir: str,
    prefix: str = "proton_decoherence"
):
    plt.figure(figsize=(6, 4))
    x = np.arange(len(band_names))
    plt.bar(x, gammas)
    plt.xticks(x, band_names)
    plt.ylabel("Decoherence rate γ_band")
    plt.title("Assigned decoherence rates per wavelet band")
    plt.grid(axis="y", alpha=0.3)
    path = os.path.join(outdir, f"{prefix}_gamma_bands.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[decoherence_microscope] Saved γ-band plot to {path}")


# ---------------------------------------------------------------------
# Decoherence microscope via wavelets
# ---------------------------------------------------------------------

def assign_gamma_per_band(
    coeffs: Dict[str, np.ndarray],
    gamma_base: float,
    gamma_high: float,
) -> Dict[str, float]:
    """
    Assign decoherence rate γ_band for each wavelet band.

    Simple model:
      - Count number of 'd' characters in the band key (e.g., 'aaa', 'aad', ...).
      - γ_band = gamma_base + gamma_high * n_d

    So coarse band 'aaa' gets γ = gamma_base, fully detailed 'ddd'
    gets gamma_base + 3 * gamma_high, etc.
    """
    gamma_dict = {}
    for band in coeffs.keys():
        n_d = band.count("d")
        gamma = gamma_base + gamma_high * n_d
        gamma_dict[band] = gamma
    return gamma_dict


def simulate_decoherence_wavelet(
    density: np.ndarray,
    barycenter: Optional[Tuple[float, float, float]],
    t_max: float,
    n_times: int,
    gamma_base: float,
    gamma_high: float,
    wavelet: str,
    outdir: str,
    prefix: str = "proton_decoherence"
):
    if not HAS_PYWT:
        raise RuntimeError(
            "PyWavelets (pywt) is required for the decoherence microscope. "
            "Install with `pip install pywavelets`."
        )

    # Wavelet decomposition (1 level 3D DWT)
    coeffs0 = pywt.dwtn(density, wavelet=wavelet, axes=(0, 1, 2))
    bands = sorted(coeffs0.keys())
    gammas = assign_gamma_per_band(coeffs0, gamma_base, gamma_high)

    # Plot the γ assignments
    plot_gamma_bands(
        band_names=bands,
        gammas=[gammas[b] for b in bands],
        outdir=outdir,
        prefix=prefix,
    )

    # Time grid
    t_values = np.linspace(0.0, t_max, n_times)
    xy_paths = []
    radial_profiles = []

    # Precompute geometry for radial profile
    r_centers, _ = compute_radial_profile(density, barycenter=barycenter, n_bins=20)

    for t in t_values:
        # Damp each band
        coeffs_t = {}
        for band, arr in coeffs0.items():
            gamma = gammas[band]
            factor = math.exp(-gamma * t)
            coeffs_t[band] = arr * factor

        # Reconstruct density at time t
        density_t = pywt.idwtn(coeffs_t, wavelet=wavelet, axes=(0, 1, 2))

        # Ensure real and non-negative (numerical cleanup)
        density_t = np.real(density_t)
        density_t[density_t < 0] = 0.0

        # XY slice
        path_xy = plot_xy_slice(
            density_t, t_label=f"{t:.2f}", outdir=outdir, prefix=prefix
        )
        xy_paths.append(path_xy)
        print(f"[decoherence_microscope] Saved XY slice at t={t:.2f} to {path_xy}")

        # Radial profile
        _, n_of_r_t = compute_radial_profile(
            density_t, barycenter=barycenter, n_bins=len(r_centers)
        )
        radial_profiles.append(n_of_r_t)

    # Plot radial profiles vs time
    plot_radial_profiles_vs_time(
        r_centers=r_centers,
        profiles=radial_profiles,
        t_values=list(t_values),
        outdir=outdir,
        prefix=prefix,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decoherence microscope for the proton density via 3D wavelets."
    )
    parser.add_argument("--proton_script", type=str, default="proton_matrixfree.py",
                        help="Path to proton_matrixfree.py (default: proton_matrixfree.py)")

    # Proton parameters (forwarded to proton_matrixfree.py)
    parser.add_argument("--Lx", type=int, default=3, help="Lx (default 3)")
    parser.add_argument("--Ly", type=int, default=3, help="Ly (default 3)")
    parser.add_argument("--Lz", type=int, default=3, help="Lz (default 3)")
    parser.add_argument("--J_hop", type=float, default=None, help="Hopping J (optional)")
    parser.add_argument("--m", type=float, default=None, help="Mass parameter (optional)")
    parser.add_argument("--g_defrag", type=float, default=None, help="Defrag strength (optional)")
    parser.add_argument("--sigma_defrag", type=float, default=None, help="Defrag width (optional)")
    parser.add_argument("--lambda_G", type=float, default=None, help="Gauss penalty λ_G (optional)")
    parser.add_argument("--B", type=float, default=None, help="B field (optional)")
    parser.add_argument("--max_iter", type=int, default=None, help="Max eigsh iterations (optional)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")

    # Decoherence microscope parameters
    parser.add_argument("--t_max", type=float, default=2.0,
                        help="Maximum decoherence time (default 2.0)")
    parser.add_argument("--n_times", type=int, default=6,
                        help="Number of time slices (default 6)")
    parser.add_argument("--gamma_base", type=float, default=0.1,
                        help="Base decoherence rate γ_base for coarse band (default 0.1)")
    parser.add_argument("--gamma_high", type=float, default=0.3,
                        help="Additional rate per 'd' in band name (default 0.3)")
    parser.add_argument("--wavelet", type=str, default="db2",
                        help="Wavelet family for DWT (default 'db2')")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    proton_script_path = os.path.join(script_dir, args.proton_script)
    if not os.path.exists(proton_script_path):
        raise FileNotFoundError(
            f"Could not find proton script at {proton_script_path}. "
            f"Use --proton_script to point to proton_matrixfree.py."
        )

    if not HAS_PYWT:
        raise RuntimeError(
            "PyWavelets (pywt) is required. Install with `pip install pywavelets`."
        )

    # Run proton solver
    stdout_text = run_proton_matrixfree(proton_script_path, args)

    # Parse density and barycenter
    parsed = parse_proton_output(stdout_text, args.Lx, args.Ly, args.Lz)
    density = parsed["density"]
    barycenter = parsed["barycenter"]
    if barycenter is None:
        bx = 0.5 * (args.Lx - 1)
        by = 0.5 * (args.Ly - 1)
        bz = 0.5 * (args.Lz - 1)
        barycenter = (bx, by, bz)
        print(f"[decoherence_microscope] No barycenter line; "
              f"using geometric center ({bx:.3f}, {by:.3f}, {bz:.3f})")
    else:
        print(f"[decoherence_microscope] Parsed barycenter: "
              f"({barycenter[0]:.3f}, {barycenter[1]:.3f}, {barycenter[2]:.3f})")

    # Outputs dir
    outdir = ensure_outdir(script_dir, "outputs")

    # Run wavelet-based decoherence simulation
    simulate_decoherence_wavelet(
        density=density,
        barycenter=barycenter,
        t_max=args.t_max,
        n_times=args.n_times,
        gamma_base=args.gamma_base,
        gamma_high=args.gamma_high,
        wavelet=args.wavelet,
        outdir=outdir,
        prefix="proton_decoherence",
    )

    print("\n" + "=" * 70)
    print("DECOHERENCE MICROSCOPE COMPLETE")
    print(f"Outputs in: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
