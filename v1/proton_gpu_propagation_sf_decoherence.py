#!/usr/bin/env python3
"""
proton_gpu_propagation_sf_decoherence.py

Standalone GPU "proton" model with internal structure + structure-factor
decoherence microscope.

Model:
- Complex field psi(x,y,z) on a 3D lattice.
- Hamiltonian H = T + V with:
    T = - (1 / (2 m_eff)) ∇^2
    V = V_conf(r) + V_shell(r) + V_aniso(x,y,z)
  where
    V_conf  ~ harmonic radial confinement
    V_shell ~ attractive shell at radius r0 to create internal structure
    V_aniso ~ small anisotropy to break full spherical symmetry

Ground state:
- Obtained by imaginary-time propagation on the GPU using split-step:
    psi -> exp(-dt V/2) * FFT^-1( exp(-dt T(k)) * FFT(psi) ) * exp(-dt V/2)
  with normalization at each step.

Decoherence microscope:
- Start from ground-state density n0 = |psi|^2 (normalized).
- Compute n_k(k) = FFT[n0].
- Define decoherence rate:
    gamma(k) = gamma0 + gamma2 * |k|^2
- For times t:
    n_k(k,t) = n_k(k,0) * exp(-gamma(k) * t)
    n(x,t)   = FFT^-1[n_k(k,t)]
    renormalize n(x,t) to keep total probability = 1
- Diagnostics vs t:
    - r_eff(t)
    - radial density profile n(r,t)
    - spherically-averaged S(k,t) = <|n_k|^2> over |k|
    - XY mid-z slices

Outputs:
- All PNGs saved in ./outputs/:
    proton_gpu_ground_xy.png
    proton_gpu_ground_radial.png
    proton_gpu_Sk_t*.png (via combined plot)
    proton_gpu_slice_xy_t*.png
    proton_gpu_radial_vs_t.png
    proton_gpu_r_eff_vs_t.png
    proton_gpu_Sk_vs_t.png
    proton_gpu_gamma_hist.png
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
# Utility: ensure output directory
# ---------------------------------------------------------------------

def ensure_outdir(base_dir: str, subdir: str = "outputs") -> str:
    out_path = os.path.join(base_dir, subdir)
    os.makedirs(out_path, exist_ok=True)
    return out_path


# ---------------------------------------------------------------------
# Geometry helpers: r_eff and radial profile (CPU side, NumPy)
# ---------------------------------------------------------------------

def compute_r_eff(density: np.ndarray) -> float:
    """
    Compute RMS radius relative to center-of-mass.

    density.shape = (Nx, Ny, Nz), indices interpreted as lattice coords.
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
    n_bins: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Azimuthally-averaged radial profile n(r).

    Returns:
      r_centers: (n_bins,)
      n_of_r:   (n_bins,)
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


# ---------------------------------------------------------------------
# k-space helpers for structure factor (CPU side)
# ---------------------------------------------------------------------

def compute_kgrid(Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """
    Compute |k| for each FFT mode, in lattice units^-1.

    Shape: (Nx, Ny, Nz).
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    return Kmag


def radial_structure_factor(
    S_k: np.ndarray,
    Kmag: np.ndarray,
    n_bins: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spherically-averaged structure factor S(k).

    S_k:   |n_k|^2, shape (Nx, Ny, Nz)
    Kmag:  |k| for each mode, same shape

    Returns:
      k_centers: (n_bins,)
      S_of_k:   (n_bins,)
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
# Potential and kinetic energy on GPU
# ---------------------------------------------------------------------

def build_coordinate_grid_gpu(Nx: int, Ny: int, Nz: int) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Build centered coordinate grid on GPU.

    x, y, z in lattice units, centered at 0.
    """
    x = cp.arange(Nx) - 0.5 * (Nx - 1)
    y = cp.arange(Ny) - 0.5 * (Ny - 1)
    z = cp.arange(Nz) - 0.5 * (Nz - 1)
    X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


def build_potential_gpu(
    Nx: int,
    Ny: int,
    Nz: int,
    omega_conf: float,
    shell_depth: float,
    shell_r0: float,
    shell_sigma: float,
    aniso_eps: float,
) -> cp.ndarray:
    """
    Build confining + shell + anisotropy potential V(x,y,z) on GPU.
    """
    X, Y, Z = build_coordinate_grid_gpu(Nx, Ny, Nz)
    R = cp.sqrt(X**2 + Y**2 + Z**2)

    # Harmonic confining potential
    V_conf = 0.5 * (omega_conf**2) * (R**2)

    # Shell-forming attractive potential
    # Negative Gaussian ring at radius shell_r0
    V_shell = -shell_depth * cp.exp(-0.5 * ((R - shell_r0) / shell_sigma) ** 2)

    # Small anisotropy to break symmetry: quadrupole-like
    V_aniso = aniso_eps * (X**2 - Y**2) / (1.0 + R**2)

    V = V_conf + V_shell + V_aniso
    return V


def build_kinetic_factor_gpu(
    Nx: int,
    Ny: int,
    Nz: int,
    dt: float,
    m_eff: float,
) -> cp.ndarray:
    """
    Build exp(-dt * T(k)) factor in k-space for split-step imaginary-time.

    T(k) = (|k|^2) / (2 m_eff)
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    Kmag2 = KX**2 + KY**2 + KZ**2
    T_k_np = 0.5 * Kmag2 / m_eff
    T_factor_np = np.exp(-dt * T_k_np)
    return cp.asarray(T_factor_np, dtype=cp.complex128)


# ---------------------------------------------------------------------
# Imaginary-time propagation (GPU, split-step)
# ---------------------------------------------------------------------

def imaginary_time_ground_state_gpu(
    Nx: int,
    Ny: int,
    Nz: int,
    dt: float,
    n_steps: int,
    omega_conf: float,
    shell_depth: float,
    shell_r0: float,
    shell_sigma: float,
    aniso_eps: float,
    m_eff: float,
    outdir: str,
) -> Tuple[np.ndarray, float]:
    """
    Find approximate ground state psi(x,y,z) via imaginary-time evolution
    on GPU using split-step method.

    Returns:
      density: |psi|^2 (NumPy, normalized)
      r_eff  : effective radius of the ground state
    """
    # Potential and kinetic factors
    V = build_potential_gpu(Nx, Ny, Nz,
                            omega_conf, shell_depth, shell_r0,
                            shell_sigma, aniso_eps)
    V_half_factor = cp.exp(-0.5 * dt * V)  # real, but exponent in imaginary time

    T_factor = build_kinetic_factor_gpu(Nx, Ny, Nz, dt, m_eff)

    # Initial psi: random complex + small bias toward center
    rng = cp.random.default_rng()
    psi = (rng.standard_normal((Nx, Ny, Nz)) +
           1j * rng.standard_normal((Nx, Ny, Nz)))
    psi *= cp.exp(-0.5 * (cp.sqrt((build_coordinate_grid_gpu(Nx, Ny, Nz)[0]**2 +
                                  build_coordinate_grid_gpu(Nx, Ny, Nz)[1]**2 +
                                  build_coordinate_grid_gpu(Nx, Ny, Nz)[2]**2))))

    # Normalize
    norm = cp.sqrt(cp.sum(cp.abs(psi)**2))
    psi /= norm

    print("Imaginary-time propagation (GPU):")
    print(f"  Grid: {Nx} x {Ny} x {Nz}")
    print(f"  dt  = {dt}, steps = {n_steps}")
    print(f"  omega_conf={omega_conf}, shell_depth={shell_depth}, "
          f"r0={shell_r0}, sigma={shell_sigma}, aniso_eps={aniso_eps}, m_eff={m_eff}")
    print("-" * 70)

    for step in range(1, n_steps + 1):
        # Half potential
        psi *= V_half_factor

        # Kinetic via FFT
        psi_k = cp.fft.fftn(psi)
        psi_k *= T_factor
        psi = cp.fft.ifftn(psi_k)

        # Half potential
        psi *= V_half_factor

        # Normalize
        norm = cp.sqrt(cp.sum(cp.abs(psi)**2))
        psi /= norm

        if step % max(1, n_steps // 10) == 0:
            # quick energy diagnostic
            # E ≈ <psi|T+V|psi>
            psi_k = cp.fft.fftn(psi)
            # T contribution in k space
            # (reuse T_factor = exp(-dt*T), so T ~ -ln(T_factor)/dt)
            T_k = -cp.log(T_factor).real / dt
            Tpsi_k = T_k * psi_k
            Tpsi = cp.fft.ifftn(Tpsi_k)
            E_T = cp.real(cp.sum(cp.conj(psi) * Tpsi))

            Vpsi = V * psi
            E_V = cp.real(cp.sum(cp.conj(psi) * Vpsi))

            E_total = float(E_T + E_V)
            print(f"  step {step:6d}/{n_steps}: E ≈ {E_total: .6f}")

    # Transfer back to CPU
    psi_np = cp.asnumpy(psi)
    density = np.abs(psi_np)**2
    total = density.sum()
    if total > 0:
        density /= total

    # r_eff
    r_eff = compute_r_eff(density)

    # Save basic ground-state visuals
    # XY mid-z slice
    Nz_mid = Nz // 2
    plt.figure(figsize=(5, 4))
    plt.imshow(density[:, :, Nz_mid].T, origin="lower", interpolation="nearest")
    plt.colorbar(label="n(x,y,z_mid)")
    plt.title("Ground-state density (XY slice at mid-z)")
    plt.xlabel("x")
    plt.ylabel("y")
    path_xy = os.path.join(outdir, "proton_gpu_ground_xy.png")
    plt.savefig(path_xy, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ground_state] Saved XY slice to {path_xy}")

    # Radial profile
    r_centers, n_of_r = compute_radial_profile(density, n_bins=32)
    plt.figure(figsize=(6, 4))
    plt.plot(r_centers, n_of_r, marker="o")
    plt.xlabel("Radius r (lattice units)")
    plt.ylabel("Average density n(r)")
    plt.title("Ground-state radial density profile")
    plt.grid(True, alpha=0.3)
    path_rad = os.path.join(outdir, "proton_gpu_ground_radial.png")
    plt.savefig(path_rad, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ground_state] Saved radial profile to {path_rad}")

    print(f"[ground_state] r_eff ≈ {r_eff:.4f}")
    return density, r_eff


# ---------------------------------------------------------------------
# Structure-factor decoherence microscope (GPU)
# ---------------------------------------------------------------------

def decoherence_microscope_sf_gpu(
    density: np.ndarray,
    t_max: float,
    n_times: int,
    gamma0: float,
    gamma2: float,
    outdir: str,
):
    """
    Run structure-factor decoherence microscope on GPU:

    - density: |psi|^2 on CPU (NumPy), normalized.
    - In k-space: n_k(t) = n_k(0) * exp(-gamma(k) * t),
      gamma(k) = gamma0 + gamma2 |k|^2.
    """
    Nx, Ny, Nz = density.shape
    print("\nStructure-factor decoherence microscope (GPU)")
    print(f"  Grid: {Nx} x {Ny} x {Nz}")
    print(f"  gamma0={gamma0}, gamma2={gamma2}, t_max={t_max}, n_times={n_times}")
    print("-" * 70)

    # k-grid & gamma(k) (CPU + GPU)
    Kmag_np = compute_kgrid(Nx, Ny, Nz)
    gamma_k_np = gamma0 + gamma2 * (Kmag_np**2)
    gamma_k_cp = cp.asarray(gamma_k_np, dtype=cp.float64)

    # Save gamma histogram
    plt.figure(figsize=(5, 4))
    plt.hist(gamma_k_np.flatten(), bins=40)
    plt.xlabel("gamma(k)")
    plt.ylabel("Count")
    plt.title("Distribution of decoherence rates gamma(k)")
    plt.grid(True, alpha=0.3)
    path_gamma = os.path.join(outdir, "proton_gpu_gamma_hist.png")
    plt.savefig(path_gamma, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[decoherence] Saved gamma(k) histogram to {path_gamma}")

    # Initial n_k on GPU
    density_cp = cp.asarray(density, dtype=cp.float64)
    n_k0_cp = cp.fft.fftn(density_cp)

    # Baseline S(k) radial profile from t=0
    S0_k_np = np.abs(cp.asnumpy(n_k0_cp))**2
    k_centers_ref, S0_of_k = radial_structure_factor(S0_k_np, Kmag_np, n_bins=32)

    t_values = np.linspace(0.0, t_max, n_times)
    r_eff_list: List[float] = []
    radial_profiles: List[np.ndarray] = []
    Sk_profiles: List[np.ndarray] = []

    for t in t_values:
        print(f"[decoherence] t={t:.3f}")
        factor_cp = cp.exp(-gamma_k_cp * t)
        n_k_t_cp = n_k0_cp * factor_cp

        # S(k,t) on CPU for diagnostics
        S_k_t_np = np.abs(cp.asnumpy(n_k_t_cp))**2
        _, S_of_k = radial_structure_factor(S_k_t_np, Kmag_np, n_bins=len(k_centers_ref))
        Sk_profiles.append(S_of_k)

        # Back to real space
        density_t_cp = cp.fft.ifftn(n_k_t_cp)
        density_t_np = np.real(cp.asnumpy(density_t_cp))
        density_t_np[density_t_np < 0] = 0.0
        tot = density_t_np.sum()
        if tot > 0:
            density_t_np *= (1.0 / tot)

        # r_eff(t)
        r_eff_t = compute_r_eff(density_t_np)
        r_eff_list.append(r_eff_t)

        # radial profile
        r_centers, n_of_r = compute_radial_profile(density_t_np, n_bins=32)
        radial_profiles.append(n_of_r)

        # XY slice
        Nz_mid = Nz // 2
        plt.figure(figsize=(5, 4))
        plt.imshow(density_t_np[:, :, Nz_mid].T,
                   origin="lower",
                   interpolation="nearest")
        plt.colorbar(label="n(x,y,z_mid)")
        plt.title(f"XY slice at mid-z, t={t:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        fname = f"proton_gpu_slice_xy_t{t:.2f}".replace(".", "p") + ".png"
        path_xy_t = os.path.join(outdir, fname)
        plt.savefig(path_xy_t, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[decoherence] Saved XY slice to {path_xy_t}")

    # Radial profiles vs t
    plt.figure(figsize=(6, 4.5))
    for n_of_r, t in zip(radial_profiles, t_values):
        plt.plot(r_centers, n_of_r, marker="o", label=f"t={t:.2f}")
    plt.xlabel("Radius r (lattice units)")
    plt.ylabel("Average density n(r)")
    plt.title("Radial density vs decoherence time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path_rad_t = os.path.join(outdir, "proton_gpu_radial_vs_t.png")
    plt.savefig(path_rad_t, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[decoherence] Saved radial vs t to {path_rad_t}")

    # r_eff vs t
    plt.figure(figsize=(5, 4))
    plt.plot(t_values, r_eff_list, marker="o")
    plt.xlabel("Decoherence time t")
    plt.ylabel("Effective radius r_eff")
    plt.title("r_eff vs decoherence time")
    plt.grid(True, alpha=0.3)
    path_reff_t = os.path.join(outdir, "proton_gpu_r_eff_vs_t.png")
    plt.savefig(path_reff_t, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[decoherence] Saved r_eff vs t to {path_reff_t}")

    # S(k) vs t
    plt.figure(figsize=(6, 4.5))
    plt.plot(k_centers_ref, S0_of_k, marker="o", label="t=0.00")
    for S_of_k, t in zip(Sk_profiles, t_values):
        if t == 0.0:
            continue
        plt.plot(k_centers_ref, S_of_k, marker="o", label=f"t={t:.2f}")
    plt.xlabel("|k| (lattice units^-1)")
    plt.ylabel("S(k)")
    plt.title("Structure factor S(k) vs decoherence time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path_Sk_t = os.path.join(outdir, "proton_gpu_Sk_vs_t.png")
    plt.savefig(path_Sk_t, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[decoherence] Saved S(k) vs t to {path_Sk_t}")

    print("\n[decoherence] Done.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global CUPY_IMPORT_ERROR

    parser = argparse.ArgumentParser(
        description="GPU proton model with internal structure and SF decoherence microscope."
    )
    # Grid
    parser.add_argument("--Nx", type=int, default=32, help="Grid size in x (default 32)")
    parser.add_argument("--Ny", type=int, default=32, help="Grid size in y (default 32)")
    parser.add_argument("--Nz", type=int, default=32, help="Grid size in z (default 32)")

    # Imaginary-time propagation
    parser.add_argument("--dt", type=float, default=0.02, help="Imaginary time step (default 0.02)")
    parser.add_argument("--n_steps", type=int, default=3000, help="Number of imaginary-time steps (default 3000)")
    parser.add_argument("--m_eff", type=float, default=1.0, help="Effective mass in kinetic term (default 1.0)")

    # Potential parameters
    parser.add_argument("--omega_conf", type=float, default=0.15, help="Harmonic confining strength (default 0.15)")
    parser.add_argument("--shell_depth", type=float, default=4.0, help="Depth of shell potential (default 4.0)")
    parser.add_argument("--shell_r0", type=float, default=4.0, help="Radius of shell (lattice units, default 4.0)")
    parser.add_argument("--shell_sigma", type=float, default=1.5, help="Width of shell (default 1.5)")
    parser.add_argument("--aniso_eps", type=float, default=0.3, help="Anisotropy strength (default 0.3)")

    # Decoherence parameters
    parser.add_argument("--t_max", type=float, default=3.0, help="Max decoherence time (default 3.0)")
    parser.add_argument("--n_times", type=int, default=7, help="Number of decoherence time slices (default 7)")
    parser.add_argument("--gamma0", type=float, default=0.05, help="Base decoherence rate gamma0 (default 0.05)")
    parser.add_argument("--gamma2", type=float, default=0.8, help="Coefficient of |k|^2 in gamma(k) (default 0.8)")

    args = parser.parse_args()

    if CUPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CuPy is required for this script.\n"
            f"Import error was: {CUPY_IMPORT_ERROR}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = ensure_outdir(script_dir, "outputs")

    # 1) Ground state via imaginary-time propagation
    density, r_eff = imaginary_time_ground_state_gpu(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        dt=args.dt,
        n_steps=args.n_steps,
        omega_conf=args.omega_conf,
        shell_depth=args.shell_depth,
        shell_r0=args.shell_r0,
        shell_sigma=args.shell_sigma,
        aniso_eps=args.aniso_eps,
        m_eff=args.m_eff,
        outdir=outdir,
    )

    # 2) Structure-factor decoherence microscope
    decoherence_microscope_sf_gpu(
        density=density,
        t_max=args.t_max,
        n_times=args.n_times,
        gamma0=args.gamma0,
        gamma2=args.gamma2,
        outdir=outdir,
    )

    print("\n" + "=" * 70)
    print("GPU PROTON + STRUCTURE-FACTOR DECOHERENCE MICROSCOPE COMPLETE")
    print(f"Outputs saved in: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
