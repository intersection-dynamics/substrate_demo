#!/usr/bin/env python3
"""
proton_analysis.py

High-level analysis wrapper for the matrix-free 3-quark proton simulator.

This script:
  1. Calls `proton_matrixfree.py` with the given parameters.
  2. Captures and prints its output so you still see the usual diagnostics.
  3. Parses the "Site occupancy" block into a 3D density array n[x,y,z].
  4. Produces:
       - mid-plane density slices (XY, XZ, YZ) as PNGs
       - a radial density profile n(r) vs r as a PNG
       - a 3D wavelet band-power spectrum (if `pywt` is available) as a PNG
  5. Saves all images under ./outputs/

Usage example (Windows friendly):

    python proton_analysis.py ^
      --Lx 4 --Ly 4 --Lz 4 ^
      --g_defrag 2.0 --sigma_defrag 0.7 ^
      --lambda_G 5.0 --max_iter 2000 --seed 42

This assumes `proton_matrixfree.py` is in the same directory and supports
the CLI flags:
  --Lx, --Ly, --Lz, --J_hop, --m, --g_defrag, --sigma_defrag,
  --lambda_G, --B, --max_iter, --seed
"""

import argparse
import os
import re
import subprocess
import math
from typing import Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# Optional wavelet support
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


# ---------------------------------------------------------------------
# Parsing proton_matrixfree.py output
# ---------------------------------------------------------------------

OCCUPANCY_LINE_RE = re.compile(
    r"r=\s*(\d+)\s+\(x,y,z\)=\((\d+),(\d+),(\d+)\)\s*:\s*n\(r\)\s*≈\s*([0-9.]+)"
)

BARYCENTER_LINE_RE = re.compile(
    r"Barycenter \(x,y,z\): \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)"
)


def run_proton_matrixfree(
    script_path: str,
    args_ns: argparse.Namespace
) -> str:
    """
    Call proton_matrixfree.py as a subprocess, capture stdout, and return it.

    Also prints the stdout back to this process's stdout so the user sees
    the normal diagnostics.

    IMPORTANT: we force PYTHONIOENCODING=utf-8 in the child process so
    unicode characters like '≈' do not cause a UnicodeEncodeError under
    cp1252 on Windows.
    """
    cmd = ["python", script_path]

    # Map Namespace to CLI flags
    def add_flag(flag: str, value: Optional[float]):
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

    # Force UTF-8 for child process' stdout/stderr to handle '≈' cleanly
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

    # Echo stdout and stderr so it looks normal to the user
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

      - density[nx, ny, nz]
      - barycenter (if present)

    Assumes the "Site occupancy" block is in the form:

      Site occupancy (normalized to 3 quarks total):
        r=  0 (x,y,z)=(0,0,0) : n(r) ≈ 0.087485
        ...

    Returns dict with keys:
      "density": np.ndarray of shape (Lx, Ly, Lz)
      "barycenter": (bx, by, bz) in lattice coordinates (float) or None
    """
    density = np.zeros((Lx, Ly, Lz), dtype=float)
    barycenter = None

    # Parse barycenter if available
    for line in text.splitlines():
        m = BARYCENTER_LINE_RE.search(line)
        if m:
            bx, by, bz = map(float, m.groups())
            barycenter = (bx, by, bz)
            break

    # Parse occupancy lines
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
                f"Parsed occupancy (x,y,z)=({x},{y},{z}) out of bounds "
                f"for L=({Lx},{Ly},{Lz})"
            )
        density[x, y, z] = n_r
        count += 1

    if count == 0:
        raise ValueError("Could not find any site occupancy lines in output.")

    total_norm = density.sum()
    print(f"\n[proton_analysis] Parsed {count} sites, total density sum = {total_norm:.6f}")
    return {"density": density, "barycenter": barycenter}


# ---------------------------------------------------------------------
# Analysis: slices, radial profile, wavelets
# ---------------------------------------------------------------------

def ensure_outdir(base_dir: str, outdir: str = "outputs") -> str:
    out_path = os.path.join(base_dir, outdir)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def plot_slices(
    density: np.ndarray,
    outdir: str,
    prefix: str = "proton_density"
):
    """
    Plot mid-plane slices in XY, XZ, YZ.

    density shape is (Lx, Ly, Lz).
    """
    Lx, Ly, Lz = density.shape
    cx = Lx // 2
    cy = Ly // 2
    cz = Lz // 2

    # XY at z=cz
    slice_xy = density[:, :, cz]
    # XZ at y=cy
    slice_xz = density[:, cy, :]
    # YZ at x=cx
    slice_yz = density[cx, :, :]

    # XY slice
    plt.figure(figsize=(4, 4))
    plt.imshow(slice_xy.T, origin="lower", interpolation="nearest")
    plt.colorbar(label="n(x,y,z_mid)")
    plt.title(f"Density slice XY (z={cz})")
    plt.xlabel("x")
    plt.ylabel("y")
    path_xy = os.path.join(outdir, f"{prefix}_slice_xy_z{cz}.png")
    plt.savefig(path_xy, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[proton_analysis] Saved XY slice to {path_xy}")

    # XZ slice
    plt.figure(figsize=(4, 4))
    plt.imshow(slice_xz.T, origin="lower", interpolation="nearest")
    plt.colorbar(label="n(x,y_mid,z)")
    plt.title(f"Density slice XZ (y={cy})")
    plt.xlabel("x")
    plt.ylabel("z")
    path_xz = os.path.join(outdir, f"{prefix}_slice_xz_y{cy}.png")
    plt.savefig(path_xz, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[proton_analysis] Saved XZ slice to {path_xz}")

    # YZ slice
    plt.figure(figsize=(4, 4))
    plt.imshow(slice_yz.T, origin="lower", interpolation="nearest")
    plt.colorbar(label="n(x_mid,y,z)")
    plt.title(f"Density slice YZ (x={cx})")
    plt.xlabel("y")
    plt.ylabel("z")
    path_yz = os.path.join(outdir, f"{prefix}_slice_yz_x{cx}.png")
    plt.savefig(path_yz, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[proton_analysis] Saved YZ slice to {path_yz}")


def compute_radial_profile(
    density: np.ndarray,
    barycenter: Optional[Tuple[float, float, float]] = None,
    n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an azimuthally-averaged radial density profile n(r).

    Returns:
      r_centers: shape (n_bins,)
      n_of_r:   shape (n_bins,)
    """
    Lx, Ly, Lz = density.shape
    xs = np.arange(Lx)
    ys = np.arange(Ly)
    zs = np.arange(Lz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    if barycenter is None:
        # Default: geometric center
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
    for r_val, n_val, idx in zip(flat_R, flat_n, inds):
        if 0 <= idx < n_bins:
            n_of_r[idx] += n_val
            counts[idx] += 1.0

    # Average density per radius bin
    mask = counts > 0
    n_of_r[mask] /= counts[mask]

    return r_centers, n_of_r


def plot_radial_profile(
    r_centers: np.ndarray,
    n_of_r: np.ndarray,
    outdir: str,
    prefix: str = "proton_density"
):
    plt.figure(figsize=(5, 4))
    plt.plot(r_centers, n_of_r, "o-")
    plt.xlabel("Radius r (lattice units)")
    plt.ylabel("Average density n(r)")
    plt.title("Radial density profile")
    plt.grid(True, alpha=0.3)
    path = os.path.join(outdir, f"{prefix}_radial_profile.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[proton_analysis] Saved radial profile to {path}")


def wavelet_analysis_3d(
    density: np.ndarray,
    outdir: str,
    prefix: str = "proton_density",
    wavelet: str = "db2"
):
    """
    Perform a single-level 3D DWT on the density and plot band powers.

    Requires pywt. If not installed, this function will return immediately.
    """
    if not HAS_PYWT:
        print("[proton_analysis] pywt not available; skipping wavelet analysis.")
        return

    coeffs = pywt.dwtn(density, wavelet=wavelet, axes=(0, 1, 2))
    # coeffs is a dict: keys like 'aaa', 'aad', 'ada', ... etc.
    band_names = []
    band_powers = []

    for band, arr in coeffs.items():
        power = float(np.sum(np.abs(arr) ** 2))
        band_names.append(band)
        band_powers.append(power)

    band_powers = np.array(band_powers)
    total_power = band_powers.sum()
    if total_power > 0:
        rel_powers = band_powers / total_power
    else:
        rel_powers = band_powers * 0.0

    # Bar plot
    plt.figure(figsize=(6, 4))
    x = np.arange(len(band_names))
    plt.bar(x, rel_powers)
    plt.xticks(x, band_names)
    plt.ylabel("Relative power")
    plt.title(f"3D wavelet band powers ({wavelet})")
    plt.grid(axis="y", alpha=0.3)
    path = os.path.join(outdir, f"{prefix}_wavelet_bands.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[proton_analysis] Saved wavelet band-power plot to {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run proton_matrixfree and analyze the resulting "
                    "3-quark density (slices, radial profile, wavelets)."
    )
    parser.add_argument("--proton_script", type=str, default="proton_matrixfree.py",
                        help="Path to proton_matrixfree.py (default: proton_matrixfree.py)")

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

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    proton_script_path = os.path.join(script_dir, args.proton_script)
    if not os.path.exists(proton_script_path):
        raise FileNotFoundError(
            f"Could not find proton script at {proton_script_path}. "
            f"Use --proton_script to point to proton_matrixfree.py."
        )

    # Run the underlying proton solver and capture its text output
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
        print(f"[proton_analysis] No barycenter line found, "
              f"using geometric center ({bx:.3f}, {by:.3f}, {bz:.3f})")
    else:
        print(f"[proton_analysis] Parsed barycenter: "
              f"({barycenter[0]:.3f}, {barycenter[1]:.3f}, {barycenter[2]:.3f})")

    # Output directory
    outdir = ensure_outdir(script_dir, "outputs")

    # 1) Density slices
    plot_slices(density, outdir, prefix="proton_density")

    # 2) Radial profile
    r_centers, n_of_r = compute_radial_profile(density, barycenter=barycenter, n_bins=20)
    plot_radial_profile(r_centers, n_of_r, outdir, prefix="proton_density")

    # 3) Wavelet analysis
    wavelet_analysis_3d(density, outdir, prefix="proton_density", wavelet="db2")

    print("\n" + "=" * 70)
    print("PROTON ANALYSIS COMPLETE")
    print(f"Outputs in: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
