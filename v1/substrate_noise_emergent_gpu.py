#!/usr/bin/env python3
"""
substrate_noise_emergent_gpu.py

GPU "substrate" toy: start from noise and let a constrained energy functional
(kinetic + local amplitude + nonlocal defrag) drive imaginary-time evolution
to whatever self-organized bound structure it prefers.

Field:
    psi(x,y,z) : complex amplitude on a 3D lattice (Nx,Ny,Nz).

Energy functional (schematic):
    E[psi] = ∫ d^3x [
        (1/(2 m_eff)) |∇psi|^2               (kinetic smoothness)
      + alpha |psi|^2                        (local mass / amplitude cost)
      + beta |psi|^4                         (local nonlinearity)
      + g_defrag |psi|^2 (G_sigma * |psi|^2) (nonlocal defrag/self-attraction)
      + V_conf(x) |psi|^2                    (optional weak confinement)
    ]

Imaginary-time propagation (split-step, on GPU):
    1) psi -> exp(-dt * V_eff/2) psi
    2) psi_k = FFT[psi]
       psi_k -> exp(-dt * T(k)) psi_k
       psi   = IFFT[psi_k]
    3) psi -> exp(-dt * V_eff/2) psi
    4) normalize

V_eff depends on density rho = |psi|^2 via:
    V_eff = alpha + beta * rho + g_defrag * (G_sigma * rho) + V_conf

Nonlocal defrag potential:
    G_sigma is a 3D Gaussian of width sigma_defrag, implemented via FFT
    convolution: (G * rho)(x) = IFFT[ FFT[G] * FFT[rho] ].

We start from random complex psi, normalized, and iterate many steps.
At the end we examine:

    - emergent density n(x,y,z) = |psi|^2
    - radial profile n(r)
    - effective radius r_eff
    - structure factor S(k) = |FFT[n]|^2
    - XY slices at selected iteration times

Outputs saved in ./outputs/:

    emergent_xy_stepXXXX.png
    emergent_xy_final.png
    emergent_radial_final.png
    emergent_Sk_final.png
    emergent_r_eff_vs_step.png
"""

import argparse
import os
import math
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except ImportError as e:
    cp = None
    CUPY_IMPORT_ERROR = e
else:
    CUPY_IMPORT_ERROR = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_outdir(base_dir: str, subdir: str = "outputs") -> str:
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def compute_r_eff(density: np.ndarray) -> float:
    """
    RMS radius relative to center-of-mass.

    density.shape = (Nx,Ny,Nz).
    Lattice coordinates: integer indices.
    """
    Nx, Ny, Nz = density.shape
    xs = np.arange(Nx)
    ys = np.arange(Ny)
    zs = np.arange(Nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    total = density.sum()
    if total <= 0:
        return 0.0

    x_cm = float((density * X).sum() / total)
    y_cm = float((density * Y).sum() / total)
    z_cm = float((density * Z).sum() / total)

    R2 = (X - x_cm) ** 2 + (Y - y_cm) ** 2 + (Z - z_cm) ** 2
    r_eff2 = float((density * R2).sum() / total)
    return math.sqrt(r_eff2)


def compute_radial_profile(
    density: np.ndarray,
    n_bins: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherically averaged radial density profile n(r).
    """
    Nx, Ny, Nz = density.shape
    xs = np.arange(Nx)
    ys = np.arange(Ny)
    zs = np.arange(Nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    total = density.sum()
    if total <= 0:
        return np.zeros(n_bins), np.zeros(n_bins)

    x_cm = float((density * X).sum() / total)
    y_cm = float((density * Y).sum() / total)
    z_cm = float((density * Z).sum() / total)

    R = np.sqrt((X - x_cm) ** 2 + (Y - y_cm) ** 2 + (Z - z_cm) ** 2)
    r_max = float(R.max())
    if r_max <= 0:
        return np.zeros(n_bins), np.zeros(n_bins)

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


def compute_kgrid(Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """
    |k| for each FFT mode, in lattice units^-1.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    return np.sqrt(KX**2 + KY**2 + KZ**2)


def radial_structure_factor(
    S_k: np.ndarray,
    Kmag: np.ndarray,
    n_bins: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherically-averaged structure factor S(k) = < |n_k|^2 >_{|k|∈bin}.
    """
    k_max = float(Kmag.max())
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
# GPU building blocks
# ---------------------------------------------------------------------

def build_coordinate_grid_gpu(
    Nx: int,
    Ny: int,
    Nz: int
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Build centered coordinate grid on GPU: x,y,z in lattice units, center at 0.
    Also returns radius R.
    """
    x = cp.arange(Nx) - 0.5 * (Nx - 1)
    y = cp.arange(Ny) - 0.5 * (Ny - 1)
    z = cp.arange(Nz) - 0.5 * (Nz - 1)
    X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
    R = cp.sqrt(X**2 + Y**2 + Z**2)
    return X, Y, Z, R


def build_conf_potential_gpu(
    R: cp.ndarray,
    omega_conf: float
) -> cp.ndarray:
    """
    Weak harmonic confining potential to keep stuff near the center.
    """
    if omega_conf <= 0:
        return cp.zeros_like(R)
    return 0.5 * (omega_conf**2) * (R**2)


def build_kinetic_factor_gpu(
    Nx: int,
    Ny: int,
    Nz: int,
    dt: float,
    m_eff: float
) -> cp.ndarray:
    """
    exp(-dt * T(k)) in k-space for imaginary-time split-step.

    T(k) = |k|^2 / (2 m_eff)
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    T_k = 0.5 * K2 / m_eff
    T_factor = np.exp(-dt * T_k)
    return cp.asarray(T_factor, dtype=cp.complex128)


def build_gaussian_kernel_kspace_gpu(
    Nx: int,
    Ny: int,
    Nz: int,
    sigma: float
) -> cp.ndarray:
    """
    FFT of a 3D Gaussian kernel G(r) ~ exp(-r^2/(2 sigma^2)) on the lattice.

    We build G on CPU, FFT it on GPU, and return G_k (complex).
    """
    x = np.arange(Nx) - 0.5 * (Nx - 1)
    y = np.arange(Ny) - 0.5 * (Ny - 1)
    z = np.arange(Nz) - 0.5 * (Nz - 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R2 = X**2 + Y**2 + Z**2

    G = np.exp(-0.5 * R2 / (sigma**2))
    # normalize kernel so that sum G = 1 (so it acts like an averaging kernel)
    G /= G.sum()

    G_cp = cp.asarray(G, dtype=cp.float64)
    G_k_cp = cp.fft.fftn(G_cp)
    return G_k_cp


# ---------------------------------------------------------------------
# Imaginary-time evolution from noise under constraints
# ---------------------------------------------------------------------

def evolve_from_noise_gpu(
    Nx: int,
    Ny: int,
    Nz: int,
    dt: float,
    n_steps: int,
    m_eff: float,
    alpha: float,
    beta: float,
    g_defrag: float,
    sigma_defrag: float,
    omega_conf: float,
    snapshot_every: int,
    outdir: str,
) -> Tuple[np.ndarray, List[float]]:
    """
    Start with random complex psi on GPU and evolve in imaginary time
    under:

        T(k) = |k|^2/(2 m_eff)
        V_eff(r) = alpha + beta * rho(r)
                   + g_defrag * (G_sigma * rho)(r)
                   + V_conf(r)

    where G_sigma is a Gaussian kernel of width sigma_defrag.

    Returns:
        density_final (NumPy)
        r_eff_history (list of floats)
    """

    print("Substrate noise evolution (GPU)")
    print(f"  Grid: {Nx} x {Ny} x {Nz}")
    print(f"  dt          = {dt}")
    print(f"  n_steps     = {n_steps}")
    print(f"  m_eff       = {m_eff}")
    print(f"  alpha       = {alpha}")
    print(f"  beta        = {beta}")
    print(f"  g_defrag    = {g_defrag}")
    print(f"  sigma_defrag= {sigma_defrag}")
    print(f"  omega_conf  = {omega_conf}")
    print(f"  snapshot_every = {snapshot_every}")
    print("-" * 70)

    # Coordinates + confining potential
    X, Y, Z, R = build_coordinate_grid_gpu(Nx, Ny, Nz)
    V_conf_cp = build_conf_potential_gpu(R, omega_conf)

    # Kinetic factor in k-space
    T_factor_cp = build_kinetic_factor_gpu(Nx, Ny, Nz, dt, m_eff)

    # Gaussian kernel in k-space for defrag convolution
    G_k_cp = build_gaussian_kernel_kspace_gpu(Nx, Ny, Nz, sigma_defrag)

    # Initial psi: pure complex noise, slightly smoothed by a Gaussian envelope
    rng = cp.random.default_rng()
    psi = (rng.standard_normal((Nx, Ny, Nz)) +
           1j * rng.standard_normal((Nx, Ny, Nz)))
    psi *= cp.exp(-0.5 * (R / (0.5 * Nx))**2)

    # Normalize
    norm = cp.sqrt(cp.sum(cp.abs(psi)**2))
    psi /= norm

    r_eff_history: List[float] = []

    # Main loop
    for step in range(1, n_steps + 1):
        # Compute density rho
        rho = cp.abs(psi)**2

        # Nonlocal defrag potential: conv = G * rho
        rho_k = cp.fft.fftn(rho)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real  # real

        # Effective potential V_eff
        # Note alpha is just a constant shift; it doesn't change spatial shape,
        # but it affects normalization speed, so keep it moderate.
        V_eff = alpha + beta * rho + g_defrag * conv_cp + V_conf_cp

        # Split-step: half potential
        psi *= cp.exp(-0.5 * dt * V_eff)

        # Kinetic via FFT
        psi_k = cp.fft.fftn(psi)
        psi_k *= T_factor_cp
        psi = cp.fft.ifftn(psi_k)

        # Recompute density for second half-step potential (to keep it symmetric)
        rho = cp.abs(psi)**2
        rho_k = cp.fft.fftn(rho)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real
        V_eff = alpha + beta * rho + g_defrag * conv_cp + V_conf_cp

        psi *= cp.exp(-0.5 * dt * V_eff)

        # Normalize
        norm = cp.sqrt(cp.sum(cp.abs(psi)**2))
        psi /= norm

        # Occasionally record r_eff & snapshots
        if step % max(1, n_steps // 20) == 0 or step == 1:
            psi_np = cp.asnumpy(psi)
            density_np = np.abs(psi_np)**2
            density_np /= density_np.sum()
            r_eff = compute_r_eff(density_np)
            r_eff_history.append(r_eff)
            print(f"  step {step:6d}/{n_steps}  r_eff ≈ {r_eff:.4f}")

        if snapshot_every > 0 and step % snapshot_every == 0:
            psi_np = cp.asnumpy(psi)
            density_np = np.abs(psi_np)**2
            density_np /= density_np.sum()
            Nz_mid = Nz // 2
            plt.figure(figsize=(5, 4))
            plt.imshow(density_np[:, :, Nz_mid].T,
                       origin="lower",
                       interpolation="nearest")
            plt.colorbar(label="n(x,y,z_mid)")
            plt.title(f"XY slice (noise evolution) step={step}")
            plt.xlabel("x")
            plt.ylabel("y")
            fname = f"emergent_xy_step{step:05d}.png"
            path = os.path.join(outdir, fname)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  saved snapshot {path}")

    # Final state to CPU
    psi_np = cp.asnumpy(psi)
    density_np = np.abs(psi_np)**2
    density_np /= density_np.sum()

    return density_np, r_eff_history


# ---------------------------------------------------------------------
# Diagnostics for final emergent structure
# ---------------------------------------------------------------------

def analyze_final_structure(
    density: np.ndarray,
    r_eff_history: List[float],
    outdir: str
):
    Nx, Ny, Nz = density.shape
    Nz_mid = Nz // 2

    # Final XY slice
    plt.figure(figsize=(5, 4))
    plt.imshow(density[:, :, Nz_mid].T,
               origin="lower",
               interpolation="nearest")
    plt.colorbar(label="n(x,y,z_mid)")
    plt.title("Emergent structure: final XY slice at mid-z")
    plt.xlabel("x")
    plt.ylabel("y")
    path_xy = os.path.join(outdir, "emergent_xy_final.png")
    plt.savefig(path_xy, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final XY slice to {path_xy}")

    # Radial profile
    r_centers, n_of_r = compute_radial_profile(density, n_bins=40)
    plt.figure(figsize=(6, 4))
    plt.plot(r_centers, n_of_r, marker="o")
    plt.xlabel("Radius r (lattice units)")
    plt.ylabel("Average density n(r)")
    plt.title("Emergent structure: final radial profile")
    plt.grid(True, alpha=0.3)
    path_rad = os.path.join(outdir, "emergent_radial_final.png")
    plt.savefig(path_rad, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final radial profile to {path_rad}")

    # r_eff history
    steps = np.linspace(0, len(r_eff_history) - 1, len(r_eff_history))
    plt.figure(figsize=(5, 4))
    plt.plot(steps, r_eff_history, marker="o")
    plt.xlabel("checkpoint index")
    plt.ylabel("r_eff")
    plt.title("r_eff over imaginary-time checkpoints")
    plt.grid(True, alpha=0.3)
    path_reff = os.path.join(outdir, "emergent_r_eff_vs_step.png")
    plt.savefig(path_reff, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved r_eff history to {path_reff}")

    # Structure factor S(k)
    density_cp = cp.asarray(density, dtype=cp.float64)
    n_k_cp = cp.fft.fftn(density_cp)
    S_k_np = np.abs(cp.asnumpy(n_k_cp))**2
    Kmag_np = compute_kgrid(Nx, Ny, Nz)
    k_centers, S_of_k = radial_structure_factor(S_k_np, Kmag_np, n_bins=40)

    plt.figure(figsize=(6, 4))
    plt.plot(k_centers, S_of_k, marker="o")
    plt.xlabel("|k| (lattice units^-1)")
    plt.ylabel("S(k)")
    plt.title("Emergent structure: final S(k)")
    plt.grid(True, alpha=0.3)
    path_Sk = os.path.join(outdir, "emergent_Sk_final.png")
    plt.savefig(path_Sk, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final S(k) to {path_Sk}")

    final_r_eff = compute_r_eff(density)
    print(f"[analysis] Final r_eff ≈ {final_r_eff:.4f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global CUPY_IMPORT_ERROR

    parser = argparse.ArgumentParser(
        description="Substrate-like GPU evolution from noise under constraints."
    )

    # Grid
    parser.add_argument("--Nx", type=int, default=48, help="Grid size in x (default 48)")
    parser.add_argument("--Ny", type=int, default=48, help="Grid size in y (default 48)")
    parser.add_argument("--Nz", type=int, default=48, help="Grid size in z (default 48)")

    # Imaginary time
    parser.add_argument("--dt", type=float, default=0.01, help="Imaginary-time step (default 0.01)")
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps (default 5000)")
    parser.add_argument("--m_eff", type=float, default=1.0, help="Effective mass in kinetic term (default 1.0)")

    # Local amplitude constraints
    parser.add_argument("--alpha", type=float, default=0.2, help="Coefficient of |psi|^2 (default 0.2)")
    parser.add_argument("--beta", type=float, default=1.0, help="Coefficient of |psi|^4 (default 1.0)")

    # Nonlocal defrag/self-attraction
    parser.add_argument("--g_defrag", type=float, default=-4.0,
                        help="Strength of nonlocal defrag term (default -4.0; negative = attraction)")
    parser.add_argument("--sigma_defrag", type=float, default=3.0,
                        help="Width of Gaussian defrag kernel (default 3.0)")

    # Confinement
    parser.add_argument("--omega_conf", type=float, default=0.05,
                        help="Harmonic confining strength (default 0.05; set 0 to remove)")

    # Output control
    parser.add_argument("--snapshot_every", type=int, default=500,
                        help="Save XY snapshot every N steps (default 500; 0 = never)")

    args = parser.parse_args()

    if CUPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CuPy is required for this script.\n"
            f"Import error was: {CUPY_IMPORT_ERROR}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = ensure_outdir(script_dir, "outputs")

    # 1) Evolve noise under constraints
    density_final, r_eff_history = evolve_from_noise_gpu(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        dt=args.dt,
        n_steps=args.n_steps,
        m_eff=args.m_eff,
        alpha=args.alpha,
        beta=args.beta,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        omega_conf=args.omega_conf,
        snapshot_every=args.snapshot_every,
        outdir=outdir,
    )

    # 2) Analyze whatever structure emerged
    analyze_final_structure(density_final, r_eff_history, outdir=outdir)

    print("\n" + "=" * 70)
    print("SUBSTRATE NOISE → EMERGENT STRUCTURE SIMULATION COMPLETE")
    print(f"Outputs in: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
