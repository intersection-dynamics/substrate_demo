#!/usr/bin/env python3
"""
proton_decoherence_microscope_full.py

Decoherence microscope for the 3-quark proton toy:

  1. Runs proton_matrixfree.py to obtain the 3D density n(x,y,z).
  2. Performs a 3D wavelet transform of the density.
  3. Assigns decoherence rates γ_band that grow with spatial frequency
     (number of 'd' characters in the band key).
  4. Simulates band-wise exponential decay:
        c_band(t) = c_band(0) * exp(-γ_band t)
     and reconstructs n(x,y,z; t) via inverse 3D DWT.
  5. At each t:
        - renormalizes density so total probability is constant
        - computes effective radius r_eff(t)
        - computes radial density profile n(r, t)
        - saves XY mid-z slice as PNG
  6. Saves summary plots:
        - radial profiles vs t
        - r_eff vs t
        - γ_band bar chart

Requires:
    - proton_matrixfree.py in the same directory
    - pywavelets: pip install pywavelets
"""

import argparse
import os
import re
import subprocess
import math
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


# ---------------------------------------------------------------------
# Run proton_matrixfree with UTF-8 enforced (fixes ≈ encoding issue)
# ---------------------------------------------------------------------

def run_proton_matrixfree(script_path: str, args_ns: argparse.Namespace) -> str:
    """
    Call proton_matrixfree.py as a subprocess, capture stdout, and return it.

    Forces PYTHONIOENCODING=utf-8 so the child process can print '≈'
    without crashing on Windows cp1252 consoles.
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


# ---------------------------------------------------------------------
# Parse proton_matrixfree output: density + barycenter
# ---------------------------------------------------------------------

OCCUPANCY_LINE_RE = re.compile(
    r"r=\s*(\d+)\s+\(x,y,z\)=\((\d+),(\d+),(\d+)\)\s*:\s*n\(r\)\s*≈\s*([0-9.]+)"
)

BARYCENTER_LINE_RE = re.compile(
    r"Barycenter \(x,y,z\): \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)"
)


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

    density is shape (Lx, Ly, Lz) – we’ll transpose as needed later.
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
# Geometry helpers: r_eff and radial profile
# ---------------------------------------------------------------------

def compute_r_eff(
    density: np.ndarray,
    barycenter: Optional[Tuple[float, float, float]] = None
) -> float:
    """
    Compute RMS radius relative to center-of-mass (or given barycenter).

    density shape is (Lx, Ly, Lz) in lattice coordinates.
    """
    Lx, Ly, Lz = density.shape
    xs = np.arange(Lx)
    ys = np.arange(Ly)
    zs = np.arange(Lz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    total = density.sum()
    if total <= 0:
        return 0.0

    if barycenter is None:
        bx = float((density * X).sum() / total)
        by = float((density * Y).sum() / total)
        bz = float((density * Z).sum() / total)
    else:
        bx, by, bz = barycenter

    R2 = (X - bx) ** 2 + (Y - by) ** 2 + (Z - bz) ** 2
    r_eff2 = float((density * R2).sum() / total)
    return math.sqrt(r_eff2)


def compute_radial_profile(
    density: np.ndarray,
    barycenter: Optional[Tuple[float, float, float]] = None,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Azimuthally-averaged radial density profile n(r).

    Returns:
      r_centers: (n_bins,)
      n_of_r:   (n_bins,)
    """
    Lx, Ly, Lz = density.shape
    xs = np.arange(Lx)
    ys = np.arange(Ly)
    zs = np.arange(Lz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    total = density.sum()
    if total <= 0:
        return np.zeros(n_bins), np.zeros(n_bins)

    if barycenter is None:
        bx = float((density * X).sum() / total)
        by = float((density * Y).sum() / total)
        bz = float((density * Z).sum() / total)
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


# ---------------------------------------------------------------------
# Wavelet decoherence
# ---------------------------------------------------------------------

def assign_gamma_per_band(
    coeffs: Dict[str, np.ndarray],
    gamma_base: float,
    gamma_high: float,
) -> Dict[str, float]:
    """
    Assign decoherence rates γ_band for each wavelet band.

    Simple model:
      - Count number of 'd' characters in the band key (e.g., 'aaa', 'aad', ...).
      - γ_band = gamma_base + gamma_high * n_d
    """
    gamma_dict = {}
    for band in coeffs.keys():
        n_d = band.count("d")
        gamma = gamma_base + gamma_high * n_d
        gamma_dict[band] = gamma
    return gamma_dict


def wavelet_decohere_density(
    density: np.ndarray,
    t: float,
    gammas: Dict[str, float],
    wavelet: str = "db2",
) -> np.ndarray:
    """
    Perform a single-level 3D DWT on the density, decay each band as
    exp(-γ_band t), and reconstruct via inverse DWT.

    Uses pywt.dwtn / idwtn for multi-axis transforms.
    """
    coeffs = pywt.dwtn(density, wavelet=wavelet, axes=(0, 1, 2))
    coeffs_t = {}

    for band, arr in coeffs.items():
        gamma = gammas.get(band, 0.0)
        factor = math.exp(-gamma * t)
        coeffs_t[band] = arr * factor

    density_t = pywt.idwtn(coeffs_t, wavelet=wavelet, axes=(0, 1, 2))
    density_t = np.real(density_t)
    density_t[density_t < 0] = 0.0
    return density_t


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def ensure_outdir(base_dir: str, subdir: str = "outputs") -> str:
    out_path = os.path.join(base_dir, subdir)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def plot_gamma_bands(
    band_names: List[str],
    gammas: List[float],
    outdir: str,
    prefix: str = "proton_decoherence",
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


def plot_xy_slice(
    density: np.ndarray,
    t_label: str,
    outdir: str,
    prefix: str = "proton_decoherence",
):
    Lx, Ly, Lz = density.shape
    cz = Lz // 2
    slice_xy = density[:, :, cz]

    plt.figure(figsize=(4.5, 4))
    plt.imshow(slice_xy.T, origin="lower", interpolation="nearest")
    plt.colorbar(label="n(x,y,z_mid)")
    plt.title(f"XY slice at mid-z, t={t_label}")
    plt.xlabel("x")
    plt.ylabel("y")
    fname = f"{prefix}_slice_xy_t{t_label.replace('.', 'p')}.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[decoherence_microscope] Saved XY slice at t={t_label} to {path}")


def plot_radial_profiles_vs_time(
    r_centers: np.ndarray,
    profiles: List[np.ndarray],
    t_values: List[float],
    outdir: str,
    prefix: str = "proton_decoherence",
):
    plt.figure(figsize=(6, 4.5))
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


def plot_r_eff_vs_time(
    t_values: List[float],
    r_eff_values: List[float],
    outdir: str,
    prefix: str = "proton_decoherence",
):
    plt.figure(figsize=(5, 4))
    plt.plot(t_values, r_eff_values, marker="o")
    plt.xlabel("Decoherence time t")
    plt.ylabel("Effective radius r_eff")
    plt.title("r_eff vs decoherence time")
    plt.grid(True, alpha=0.3)
    path = os.path.join(outdir, f"{prefix}_r_eff_vs_t.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[decoherence_microscope] Saved r_eff(t) plot to {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decoherence microscope for the proton density via 3D wavelets."
    )
    parser.add_argument("--proton_script", type=str, default="proton_matrixfree.py",
                        help="Path to proton_matrixfree.py (default: proton_matrixfree.py)")

    # Proton parameters (forwarded)
    parser.add_argument("--Lx", type=int, default=4)
    parser.add_argument("--Ly", type=int, default=4)
    parser.add_argument("--Lz", type=int, default=4)
    parser.add_argument("--J_hop", type=float, default=None)
    parser.add_argument("--m", type=float, default=None)
    parser.add_argument("--g_defrag", type=float, default=2.0)
    parser.add_argument("--sigma_defrag", type=float, default=0.7)
    parser.add_argument("--lambda_G", type=float, default=5.0)
    parser.add_argument("--B", type=float, default=None)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Decoherence params
    parser.add_argument("--t_max", type=float, default=3.0)
    parser.add_argument("--n_times", type=int, default=7)
    parser.add_argument("--gamma_base", type=float, default=0.05)
    parser.add_argument("--gamma_high", type=float, default=0.8)
    parser.add_argument("--wavelet", type=str, default="db2")

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

    outdir = ensure_outdir(script_dir, "outputs")

    # 1) Run proton solver
    stdout_text = run_proton_matrixfree(proton_script_path, args)

    # 2) Parse density and barycenter
    parsed = parse_proton_output(stdout_text, args.Lx, args.Ly, args.Lz)
    density = parsed["density"]
    barycenter = parsed["barycenter"]

    if barycenter is None:
        Lx, Ly, Lz = args.Lx, args.Ly, args.Lz
        bx = 0.5 * (Lx - 1)
        by = 0.5 * (Ly - 1)
        bz = 0.5 * (Lz - 1)
        barycenter = (bx, by, bz)
        print(f"[decoherence_microscope] No barycenter line; "
              f"using geometric center ({bx:.3f}, {by:.3f}, {bz:.3f})")
    else:
        print(f"[decoherence_microscope] Parsed barycenter: "
              f"({barycenter[0]:.3f}, {barycenter[1]:.3f}, {barycenter[2]:.3f})")

    # Normalize initial density to total=1 for clarity
    total0 = density.sum()
    if total0 > 0:
        density = density / total0

    # 3) Build gamma dict from wavelet bands
    #    Use initial coeffs simply to know which bands exist
    coeffs0 = pywt.dwtn(density, wavelet=args.wavelet, axes=(0, 1, 2))
    bands = sorted(coeffs0.keys())
    gammas = assign_gamma_per_band(coeffs0, args.gamma_base, args.gamma_high)

    # Plot γ-band bar chart
    plot_gamma_bands(
        band_names=bands,
        gammas=[gammas[b] for b in bands],
        outdir=outdir,
        prefix="proton_decoherence",
    )

    # 4) Time evolution
    t_values = np.linspace(0.0, args.t_max, args.n_times)
    r_eff_values: List[float] = []
    radial_profiles: List[np.ndarray] = []
    r_centers_ref: Optional[np.ndarray] = None

    for t in t_values:
        print(f"[decoherence_microscope] Computing t={t:.3f}")
        density_t = wavelet_decohere_density(
            density=density,
            t=t,
            gammas=gammas,
            wavelet=args.wavelet,
        )

        # Renormalize
        tot = density_t.sum()
        if tot > 0:
            density_t *= (1.0 / tot)

        # r_eff
        r_eff_t = compute_r_eff(density_t, barycenter=barycenter)
        r_eff_values.append(r_eff_t)

        # radial profile
        r_centers, n_of_r = compute_radial_profile(
            density_t, barycenter=barycenter, n_bins=20
        )
        radial_profiles.append(n_of_r)
        if r_centers_ref is None:
            r_centers_ref = r_centers

        # XY slice at mid-z
        plot_xy_slice(
            density_t,
            t_label=f"{t:.2f}",
            outdir=outdir,
            prefix="proton_decoherence",
        )

    # 5) Summary plots
    if r_centers_ref is None:
        r_centers_ref = np.linspace(0.0, 1.0, 20)

    plot_radial_profiles_vs_time(
        r_centers=r_centers_ref,
        profiles=radial_profiles,
        t_values=list(t_values),
        outdir=outdir,
        prefix="proton_decoherence",
    )

    plot_r_eff_vs_time(
        t_values=list(t_values),
        r_eff_values=r_eff_values,
        outdir=outdir,
        prefix="proton_decoherence",
    )

    print("\n" + "=" * 70)
    print("DECOHERENCE MICROSCOPE COMPLETE")
    print(f"Outputs in: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
