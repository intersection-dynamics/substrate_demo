#!/usr/bin/env python3
"""
proton_sf_decoherence_gpu.py

"Structure-factor decoherence microscope" for the 3-quark proton toy.

Workflow
--------
1. Runs proton_matrixfree.py to obtain the 3D density n(x,y,z) on a small lattice.
2. Normalizes density so total probability = 1.
3. Transfers density to GPU (CuPy).
4. Computes 3D FFT n(k) and structure-factor S(k) = |n(k)|^2.
5. Defines a decoherence rate for each k-mode:
       gamma(k) = gamma0 + gamma2 * |k|^2
   where k are the usual discrete FFT wavevectors in lattice units.
6. For a set of decoherence "times" t:
       n(k,t) = n(k,0) * exp(-gamma(k) * t)
       n(x,t) = IFFT[n(k,t)]
   (all on the GPU, then converted back to NumPy for diagnostics).
7. At each t:
   - renormalizes n(x,t) to keep total probability = 1
   - computes effective radius r_eff(t)
   - computes radial density profile n(r,t)
   - computes spherically averaged S(k,t)
   - saves XY mid-z slice as PNG
8. Saves summary plots:
   - radial real-space profiles vs t
   - r_eff vs t
   - S(k) vs k at all t
   - k-space gamma(k) histogram (for sanity)

Requires:
    - proton_matrixfree.py in the same directory
    - CuPy (GPU): pip install cupy-cuda11x  (or appropriate wheel)
"""

import argparse
import os
import re
import subprocess
import math
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# CuPy for GPU work
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None  # just to avoid lint errors


# ---------------------------------------------------------------------
# Subprocess: run proton_matrixfree.py with UTF-8 (fix ≈ Unicode issue)
# ---------------------------------------------------------------------

def run_proton_matrixfree(script_path: str, args_ns: argparse.Namespace) -> str:
    """
    Call proton_matrixfree.py as a subprocess, capture stdout, and return it.

    Forces PYTHONIOENCODING=utf-8 so the child process can print '≈'
    without crashing under cp1252 on Windows.
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

      - density[x,y,z] as np.ndarray of shape (Lx, Ly, Lz)
      - barycenter (if present) as (bx,by,bz) in lattice coordinates
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
    print(f"\n[sf_decoherence] Parsed {count} sites, total density = {total:.6f}")
    return {"density": density, "barycenter": barycenter}


# ---------------------------------------------------------------------
# Geometry helpers: r_eff and radial profile in real space
# ---------------------------------------------------------------------

def compute_r_eff(
    density: np.ndarray,
    barycenter: Optional[Tuple[float, float, float]] = None
) -> float:
    """
    Compute RMS radius relative to center-of-mass (or given barycenter).

    density shape is (Lx, Ly, Lz).
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
# k-space helpers: structure-factor radial profile
# ---------------------------------------------------------------------

def compute_k_grid(Lx: int, Ly: int, Lz: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-space magnitude grid for FFT modes (NumPy, CPU side).

    Returns:
      Kmag: shape (Lx, Ly, Lz)
      kmax: scalar
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(Lx)  # in lattice units^-1
    ky = 2.0 * np.pi * np.fft.fftfreq(Ly)
    kz = 2.0 * np.pi * np.fft.fftfreq(Lz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    return Kmag, Kmag.max()


def compute_radial_Sk(
    S_k: np.ndarray,
    Kmag: np.ndarray,
    n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherically averaged structure factor S(k):

    S_k:   |n(k)|^2, shape (Lx, Ly, Lz)
    Kmag:  |k| for each mode, same shape

    Returns:
      k_centers: (n_bins,)
      S_of_k:   (n_bins,)
    """
    k_max = Kmag.max()
    if k_max <= 0:
        return np.zeros(n_bins), np.zeros(n_bins)

    bins = np.linspace(0.0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    S_of_k = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)

    flat_k = Kmag.flatten()
    flat_S = S_k.flatten()
    inds = np.digitize(flat_k, bins) - 1

    for kv, Sv, idx in zip(flat_k, flat_S, inds):
        if 0 <= idx < n_bins:
            S_of_k[idx] += Sv
            counts[idx] += 1.0

    mask = counts > 0
    S_of_k[mask] /= counts[mask]
    return k_centers, S_of_k


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def ensure_outdir(base_dir: str, subdir: str = "outputs") -> str:
    out_path = os.path.join(base_dir, subdir)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def plot_xy_slice(
    density: np.ndarray,
    t_label: str,
    outdir: str,
    prefix: str = "proton_sf",
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
    print(f"[sf_decoherence] Saved XY slice at t={t_label} to {path}")


def plot_radial_profiles_vs_time(
    r_centers: np.ndarray,
    profiles: List[np.ndarray],
    t_values: List[float],
    outdir: str,
    prefix: str = "proton_sf",
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
    print(f"[sf_decoherence] Saved radial profiles to {path}")


def plot_r_eff_vs_time(
    t_values: List[float],
    r_eff_values: List[float],
    outdir: str,
    prefix: str = "proton_sf",
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
    print(f"[sf_decoherence] Saved r_eff(t) plot to {path}")


def plot_Sk_vs_time(
    k_centers: np.ndarray,
    Sk_profiles: List[np.ndarray],
    t_values: List[float],
    outdir: str,
    prefix: str = "proton_sf",
):
    plt.figure(figsize=(6, 4.5))
    for S_of_k, t in zip(Sk_profiles, t_values):
        plt.plot(k_centers, S_of_k, marker="o", label=f"t={t:.2f}")
    plt.xlabel("|k| (lattice units^-1)")
    plt.ylabel("S(k)")
    plt.title("Structure factor S(k) vs decoherence time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(outdir, f"{prefix}_Sk_vs_t.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[sf_decoherence] Saved S(k) vs t plot to {path}")


def plot_gamma_hist(
    gamma_k: np.ndarray,
    outdir: str,
    prefix: str = "proton_sf",
):
    plt.figure(figsize=(5, 4))
    plt.hist(gamma_k.flatten(), bins=30)
    plt.xlabel("gamma(k)")
    plt.ylabel("Count")
    plt.title("Distribution of decoherence rates gamma(k)")
    plt.grid(True, alpha=0.3)
    path = os.path.join(outdir, f"{prefix}_gamma_hist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[sf_decoherence] Saved gamma(k) histogram to {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Structure-factor decoherence microscope (GPU, CuPy)."
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

    # Decoherence params in k-space
    parser.add_argument("--t_max", type=float, default=3.0,
                        help="Maximum decoherence time (default 3.0)")
    parser.add_argument("--n_times", type=int, default=7,
                        help="Number of time slices (default 7)")
    parser.add_argument("--gamma0", type=float, default=0.05,
                        help="Base decoherence rate gamma0 (default 0.05)")
    parser.add_argument("--gamma2", type=float, default=0.5,
                        help="Coefficient for |k|^2 in gamma(k) (default 0.5)")

    args = parser.parse_args()

    if not HAS_CUPY:
        raise RuntimeError(
            "CuPy is required for this script. Install with e.g. "
            "`pip install cupy-cuda11x` for an NVIDIA GPU."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    proton_script_path = os.path.join(script_dir, args.proton_script)
    if not os.path.exists(proton_script_path):
        raise FileNotFoundError(
            f"Could not find proton script at {proton_script_path}. "
            f"Use --proton_script to point to proton_matrixfree.py."
        )

    outdir = ensure_outdir(script_dir, "outputs")

    # 1) Run proton solver (CPU)
    stdout_text = run_proton_matrixfree(proton_script_path, args)

    # 2) Parse density + barycenter
    parsed = parse_proton_output(stdout_text, args.Lx, args.Ly, args.Lz)
    density = parsed["density"]
    barycenter = parsed["barycenter"]

    if barycenter is None:
        Lx, Ly, Lz = args.Lx, args.Ly, args.Lz
        bx = 0.5 * (Lx - 1)
        by = 0.5 * (Ly - 1)
        bz = 0.5 * (Lz - 1)
        barycenter = (bx, by, bz)
        print(f"[sf_decoherence] No barycenter line; "
              f"using geometric center ({bx:.3f}, {by:.3f}, {bz:.3f})")
    else:
        print(f"[sf_decoherence] Parsed barycenter: "
              f"({barycenter[0]:.3f}, {barycenter[1]:.3f}, {barycenter[2]:.3f})")

    # Normalize initial density to total=1
    total0 = density.sum()
    if total0 > 0:
        density = density / total0

    Lx, Ly, Lz = density.shape

    # 3) Build k-space grid and gamma(k) on GPU
    Kmag_np, _ = compute_k_grid(Lx, Ly, Lz)
    Kmag_cp = cp.asarray(Kmag_np)
    gamma_k_cp = args.gamma0 + args.gamma2 * (Kmag_cp ** 2)

    # Save gamma histogram (CPU)
    gamma_k_np = cp.asnumpy(gamma_k_cp)
    plot_gamma_hist(gamma_k_np, outdir=outdir, prefix="proton_sf")

    # 4) FFT of initial density on GPU
    density_cp = cp.asarray(density, dtype=cp.float64)
    n_k0_cp = cp.fft.fftn(density_cp)
    # We'll reuse Kmag_np & Kmag_cp later for S(k)

    # Precompute S(k) radial bin centers using t=0
    S0_k_np = np.abs(cp.asnumpy(n_k0_cp)) ** 2
    k_centers_ref, _ = compute_radial_Sk(S0_k_np, Kmag_np, n_bins=20)

    # 5) Time evolution
    t_values = np.linspace(0.0, args.t_max, args.n_times)
    r_eff_values: List[float] = []
    radial_profiles: List[np.ndarray] = []
    Sk_profiles: List[np.ndarray] = []

    for t in t_values:
        print(f"[sf_decoherence] Computing t={t:.3f}")
        # n(k,t) = n(k,0) * exp(-gamma_k t)
        factor_cp = cp.exp(-gamma_k_cp * t)
        n_k_t_cp = n_k0_cp * factor_cp

        # S(k,t) on CPU
        S_k_t_np = np.abs(cp.asnumpy(n_k_t_cp)) ** 2
        _, S_of_k = compute_radial_Sk(S_k_t_np, Kmag_np, n_bins=len(k_centers_ref))
        Sk_profiles.append(S_of_k)

        # Back to real space
        density_t_cp = cp.fft.ifftn(n_k_t_cp)
        density_t_np = np.real(cp.asnumpy(density_t_cp))
        density_t_np[density_t_np < 0] = 0.0

        # Renormalize
        tot = density_t_np.sum()
        if tot > 0:
            density_t_np *= (1.0 / tot)

        # r_eff(t)
        r_eff_t = compute_r_eff(density_t_np, barycenter=barycenter)
        r_eff_values.append(r_eff_t)

        # radial n(r,t)
        r_centers, n_of_r = compute_radial_profile(
            density_t_np, barycenter=barycenter, n_bins=20
        )
        radial_profiles.append(n_of_r)

        # XY slice
        plot_xy_slice(
            density_t_np,
            t_label=f"{t:.2f}",
            outdir=outdir,
            prefix="proton_sf",
        )

    # 6) Summary plots
    plot_radial_profiles_vs_time(
        r_centers=r_centers,
        profiles=radial_profiles,
        t_values=list(t_values),
        outdir=outdir,
        prefix="proton_sf",
    )

    plot_r_eff_vs_time(
        t_values=list(t_values),
        r_eff_values=r_eff_values,
        outdir=outdir,
        prefix="proton_sf",
    )

    plot_Sk_vs_time(
        k_centers=k_centers_ref,
        Sk_profiles=Sk_profiles,
        t_values=list(t_values),
        outdir=outdir,
        prefix="proton_sf",
    )

    print("\n" + "=" * 70)
    print("STRUCTURE-FACTOR DECOHERENCE MICROSCOPE COMPLETE")
    print(f"Outputs in: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
