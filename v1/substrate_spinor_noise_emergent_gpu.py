#!/usr/bin/env python3
"""
substrate_spinor_noise_emergent_gpu.py

Spinor substrate toy: start from noise and let a constrained energy functional
act on a two-component spinor field psi_s(x,y,z), s ∈ {↑, ↓}, on a 3D lattice.

- Uses CuPy for GPU acceleration.
- Imaginary-time evolution (split-step):
      psi -> exp(-dt V_eff/2) psi
             FFT -> exp(-dt T(k)) -> IFFT
             exp(-dt V_eff/2) psi
  with normalization after each step.

Energy functional (schematic):

  Let rho_up   = |psi_up|^2
      rho_dn   = |psi_dn|^2
      rho_tot  = rho_up + rho_dn
      m_z      = rho_up - rho_dn

  T(k) = |k|^2 / (2 m_eff)
  V_conf(r) = 0.5 * omega_conf^2 * r^2

  Local + nonlocal:
      V0(r) = alpha
              + beta * rho_tot(r)
              + g_defrag * (G_sigma * rho_tot)(r)
              + V_conf(r)

  Spin coupling:
      V_spin_up = +gamma_spin * m_z(r)
      V_spin_dn = -gamma_spin * m_z(r)

  So each component feels:
      V_up = V0 + V_spin_up
      V_dn = V0 + V_spin_dn

G_sigma is a 3D Gaussian kernel of width sigma_defrag, implemented via FFT
convolution: (G * rho)(x) = IFFT[ FFT[G] * FFT[rho] ].

Outputs:
    outputs/spinor_emergent_xy_final.png
    outputs/spinor_emergent_spin_xy_final.png
    outputs/spinor_emergent_radial_final.png
    outputs/spinor_emergent_spin_radial_final.png
    outputs/spinor_emergent_r_eff_vs_step.png
    outputs/spinor_emergent_Sn_final.png
    outputs/spinor_emergent_Sm_final.png

Optional comparison (if --compare_twofermion is set and substrate_engine_3d is
available):

    - runs substrate_engine_3d.run_twofermion3d_experiment on small L
    - produces:
        outputs/spinor_vs_twofermion_radial.png
        outputs/spinor_vs_twofermion_Sk.png
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


# =============================================================================
# Helpers (CPU side)
# =============================================================================

def ensure_outdir(base_dir: str, subdir: str = "outputs") -> str:
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def compute_r_eff(density: np.ndarray) -> float:
    """
    RMS radius relative to center-of-mass, using lattice indices as coordinates.
    density: shape (Nx,Ny,Nz).
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
    Spherically averaged radial profile n(r).
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
    Spherically-averaged structure factor S(k) over |k|.
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


# =============================================================================
# GPU building blocks
# =============================================================================

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
    Weak harmonic confining potential.
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
    ky = 2.0 * np.fft.fftfreq(Ny)
    kz = 2.0 * np.fft.fftfreq(Nz)
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
    FFT of a normalized 3D Gaussian kernel G_sigma on the lattice.
    """
    x = np.arange(Nx) - 0.5 * (Nx - 1)
    y = np.arange(Ny) - 0.5 * (Ny - 1)
    z = np.arange(Nz) - 0.5 * (Nz - 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R2 = X**2 + Y**2 + Z**2

    G = np.exp(-0.5 * R2 / (sigma**2))
    G /= G.sum()

    G_cp = cp.asarray(G, dtype=cp.float64)
    G_k_cp = cp.fft.fftn(G_cp)
    return G_k_cp


# =============================================================================
# Spinor evolution from noise
# =============================================================================

def evolve_spinor_from_noise_gpu(
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
    gamma_spin: float,
    snapshot_every: int,
    outdir: str,
):
    """
    Two-component spinor evolution from noise in imaginary time on the GPU.

    psi has shape (2, Nx,Ny,Nz), components [0]=up, [1]=down.

    Returns:
        density_tot_final (NumPy)
        spin_polarization_final (NumPy, m_z)
        r_eff_history (list of floats)
    """
    print("Spinor substrate evolution from noise (GPU)")
    print(f"  Grid: {Nx} x {Ny} x {Nz}")
    print(f"  dt          = {dt}")
    print(f"  n_steps     = {n_steps}")
    print(f"  m_eff       = {m_eff}")
    print(f"  alpha       = {alpha}")
    print(f"  beta        = {beta}")
    print(f"  g_defrag    = {g_defrag}")
    print(f"  sigma_defrag= {sigma_defrag}")
    print(f"  omega_conf  = {omega_conf}")
    print(f"  gamma_spin  = {gamma_spin}")
    print(f"  snapshot_every = {snapshot_every}")
    print("-" * 70)

    # Coordinates + potential
    X, Y, Z, R = build_coordinate_grid_gpu(Nx, Ny, Nz)
    V_conf_cp = build_conf_potential_gpu(R, omega_conf)

    # Kinetic factor and defrag kernel in k-space
    T_factor_cp = build_kinetic_factor_gpu(Nx, Ny, Nz, dt, m_eff)
    G_k_cp = build_gaussian_kernel_kspace_gpu(Nx, Ny, Nz, sigma_defrag)

    # Initial spinor psi: random complex for both spin components,
    # with a mild Gaussian envelope to avoid edge domination.
    rng = cp.random.default_rng()
    psi = cp.empty((2, Nx, Ny, Nz), dtype=cp.complex128)
    envelope = cp.exp(-0.5 * (R / (0.5 * Nx))**2)
    for s in range(2):
        psi_s = (rng.standard_normal((Nx, Ny, Nz)) +
                 1j * rng.standard_normal((Nx, Ny, Nz)))
        psi[s] = psi_s * envelope

    # Normalize total probability
    norm = cp.sqrt(cp.sum(cp.abs(psi)**2))
    psi /= norm

    r_eff_history: List[float] = []

    for step in range(1, n_steps + 1):
        # Densities
        rho_up = cp.abs(psi[0])**2
        rho_dn = cp.abs(psi[1])**2
        rho_tot = rho_up + rho_dn
        m_z = rho_up - rho_dn

        # Nonlocal defrag: G * rho_tot
        rho_k = cp.fft.fftn(rho_tot)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real

        # Shared potential
        V0 = alpha + beta * rho_tot + g_defrag * conv_cp + V_conf_cp

        # Spin coupling
        V_up = V0 + gamma_spin * m_z
        V_dn = V0 - gamma_spin * m_z

        # Half-step potential
        psi[0] *= cp.exp(-0.5 * dt * V_up)
        psi[1] *= cp.exp(-0.5 * dt * V_dn)

        # Kinetic via FFT (shared T_factor)
        psi_k = cp.fft.fftn(psi, axes=(1, 2, 3))
        psi_k *= T_factor_cp[None, :, :, :]
        psi = cp.fft.ifftn(psi_k, axes=(1, 2, 3))

        # Recompute densities for second half-step
        rho_up = cp.abs(psi[0])**2
        rho_dn = cp.abs(psi[1])**2
        rho_tot = rho_up + rho_dn
        m_z = rho_up - rho_dn

        rho_k = cp.fft.fftn(rho_tot)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real

        V0 = alpha + beta * rho_tot + g_defrag * conv_cp + V_conf_cp
        V_up = V0 + gamma_spin * m_z
        V_dn = V0 - gamma_spin * m_z

        psi[0] *= cp.exp(-0.5 * dt * V_up)
        psi[1] *= cp.exp(-0.5 * dt * V_dn)

        # Normalize total
        norm = cp.sqrt(cp.sum(cp.abs(psi)**2))
        psi /= norm

        # Checkpoint diagnostics
        if step % max(1, n_steps // 20) == 0 or step == 1:
            psi_np = cp.asnumpy(psi)
            density_tot_np = np.abs(psi_np[0])**2 + np.abs(psi_np[1])**2
            density_tot_np /= density_tot_np.sum()
            r_eff = compute_r_eff(density_tot_np)
            r_eff_history.append(r_eff)
            print(f"  step {step:6d}/{n_steps}  r_eff ≈ {r_eff:.4f}")

        # Snapshots of XY slices
        if snapshot_every > 0 and step % snapshot_every == 0:
            psi_np = cp.asnumpy(psi)
            density_tot_np = np.abs(psi_np[0])**2 + np.abs(psi_np[1])**2
            density_tot_np /= density_tot_np.sum()
            m_z_np = np.abs(psi_np[0])**2 - np.abs(psi_np[1])**2

            Nz_mid = density_tot_np.shape[2] // 2
            plt.figure(figsize=(5, 4))
            plt.imshow(density_tot_np[:, :, Nz_mid].T,
                       origin="lower", interpolation="nearest")
            plt.colorbar(label="n_tot(x,y,z_mid)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Spinor total density, step={step}")
            fname = f"spinor_emergent_xy_step{step:05d}.png"
            path = os.path.join(outdir, fname)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  saved snapshot {path}")

            plt.figure(figsize=(5, 4))
            plt.imshow(m_z_np[:, :, Nz_mid].T,
                       origin="lower", interpolation="nearest")
            plt.colorbar(label="m_z(x,y,z_mid)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Spin polarization m_z, step={step}")
            fname = f"spinor_emergent_spin_xy_step{step:05d}.png"
            path = os.path.join(outdir, fname)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  saved spin snapshot {path}")

    # Final state to CPU
    psi_np = cp.asnumpy(psi)
    rho_up_np = np.abs(psi_np[0])**2
    rho_dn_np = np.abs(psi_np[1])**2
    density_tot_np = rho_up_np + rho_dn_np
    total = density_tot_np.sum()
    if total > 0:
        density_tot_np /= total
    m_z_np = rho_up_np - rho_dn_np

    return density_tot_np, m_z_np, r_eff_history


# =============================================================================
# Analysis of final spinor structure
# =============================================================================

def analyze_spinor_final(
    density_tot: np.ndarray,
    m_z: np.ndarray,
    r_eff_history: List[float],
    outdir: str,
):
    Nx, Ny, Nz = density_tot.shape
    Nz_mid = Nz // 2

    # XY slice of total density
    plt.figure(figsize=(5, 4))
    plt.imshow(density_tot[:, :, Nz_mid].T,
               origin="lower", interpolation="nearest")
    plt.colorbar(label="n_tot(x,y,z_mid)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Spinor emergent: final total density (XY slice)")
    path_xy = os.path.join(outdir, "spinor_emergent_xy_final.png")
    plt.savefig(path_xy, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final XY total density to {path_xy}")

    # XY slice of spin polarization
    plt.figure(figsize=(5, 4))
    plt.imshow(m_z[:, :, Nz_mid].T,
               origin="lower", interpolation="nearest")
    plt.colorbar(label="m_z(x,y,z_mid)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Spinor emergent: final spin polarization m_z (XY slice)")
    path_spin_xy = os.path.join(outdir, "spinor_emergent_spin_xy_final.png")
    plt.savefig(path_spin_xy, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final XY spin polarization to {path_spin_xy}")

    # Radial profiles: density and spin
    r_centers, n_of_r = compute_radial_profile(density_tot, n_bins=40)
    _, m_of_r = compute_radial_profile(np.abs(m_z), n_bins=40)

    plt.figure(figsize=(6, 4))
    plt.plot(r_centers, n_of_r, marker="o")
    plt.xlabel("Radius r (lattice units)")
    plt.ylabel("Average density n_tot(r)")
    plt.title("Spinor emergent: final radial density profile")
    plt.grid(True, alpha=0.3)
    path_rad = os.path.join(outdir, "spinor_emergent_radial_final.png")
    plt.savefig(path_rad, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final radial density profile to {path_rad}")

    plt.figure(figsize=(6, 4))
    plt.plot(r_centers, m_of_r, marker="o")
    plt.xlabel("Radius r (lattice units)")
    plt.ylabel("Average |m_z(r)|")
    plt.title("Spinor emergent: final radial spin polarization")
    plt.grid(True, alpha=0.3)
    path_spin_rad = os.path.join(outdir, "spinor_emergent_spin_radial_final.png")
    plt.savefig(path_spin_rad, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final radial spin profile to {path_spin_rad}")

    # r_eff vs checkpoint index
    steps = np.arange(len(r_eff_history))
    plt.figure(figsize=(5, 4))
    plt.plot(steps, r_eff_history, marker="o")
    plt.xlabel("checkpoint index")
    plt.ylabel("r_eff")
    plt.title("Spinor emergent: r_eff over imaginary-time checkpoints")
    plt.grid(True, alpha=0.3)
    path_reff = os.path.join(outdir, "spinor_emergent_r_eff_vs_step.png")
    plt.savefig(path_reff, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved r_eff history to {path_reff}")

    # Structure factors S_n(k) and S_m(k)
    density_cp = cp.asarray(density_tot, dtype=cp.float64)
    m_z_cp = cp.asarray(m_z, dtype=cp.float64)

    n_k_cp = cp.fft.fftn(density_cp)
    m_k_cp = cp.fft.fftn(m_z_cp)

    S_n_k = np.abs(cp.asnumpy(n_k_cp))**2
    S_m_k = np.abs(cp.asnumpy(m_k_cp))**2

    Kmag = compute_kgrid(Nx, Ny, Nz)
    k_centers, S_n_of_k = radial_structure_factor(S_n_k, Kmag, n_bins=40)
    _, S_m_of_k = radial_structure_factor(S_m_k, Kmag, n_bins=40)

    plt.figure(figsize=(6, 4))
    plt.plot(k_centers, S_n_of_k, marker="o")
    plt.xlabel("|k| (lattice units^-1)")
    plt.ylabel("S_n(k)")
    plt.title("Spinor emergent: final structure factor S_n(k)")
    plt.grid(True, alpha=0.3)
    path_Sn = os.path.join(outdir, "spinor_emergent_Sn_final.png")
    plt.savefig(path_Sn, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final S_n(k) to {path_Sn}")

    plt.figure(figsize=(6, 4))
    plt.plot(k_centers, S_m_of_k, marker="o")
    plt.xlabel("|k| (lattice units^-1)")
    plt.ylabel("S_m(k)")
    plt.title("Spinor emergent: final spin structure factor S_m(k)")
    plt.grid(True, alpha=0.3)
    path_Sm = os.path.join(outdir, "spinor_emergent_Sm_final.png")
    plt.savefig(path_Sm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved final S_m(k) to {path_Sm}")

    final_r_eff = compute_r_eff(density_tot)
    print(f"[analysis] Final r_eff ≈ {final_r_eff:.4f}")


# =============================================================================
# Optional comparison to substrate_engine_3d two-fermion model
# =============================================================================

def compare_with_twofermion(
    density_spinor: np.ndarray,
    outdir: str,
    ref_L: int = 2
):
    """
    If substrate_engine_3d is available, run the two-fermion 3D model and
    compare its rho_tot(r) and S(k) to the spinor emergent density.
    """
    try:
        from substrate_engine_3d import TwoFermion3DParams, run_twofermion3d_experiment
    except Exception as e:
        print(f"[twofermion] Could not import substrate_engine_3d: {e}")
        return

    print("[twofermion] Running comparison with substrate_engine_3d...")
    params = TwoFermion3DParams(Lx=ref_L, Ly=ref_L, Lz=ref_L)
    result = run_twofermion3d_experiment(params)

    rho_tot_flat = np.array(result["rho_tot"], dtype=float)
    Lx = params.Lx
    Ly = params.Ly
    Lz = params.Lz
    rho_tot_3d = rho_tot_flat.reshape((Lx, Ly, Lz))

    # Radial profiles
    r_spinor, n_spinor = compute_radial_profile(density_spinor, n_bins=40)
    r_two, n_two = compute_radial_profile(rho_tot_3d, n_bins=40)

    plt.figure(figsize=(6, 4))
    plt.plot(r_spinor, n_spinor, marker="o", label="spinor emergent")
    plt.plot(r_two, n_two, marker="s", label="two-fermion 3D")
    plt.xlabel("Radius r (lattice units)")
    plt.ylabel("Average density n(r)")
    plt.title("Spinor vs two-fermion radial profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path_rad_cmp = os.path.join(outdir, "spinor_vs_twofermion_radial.png")
    plt.savefig(path_rad_cmp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[twofermion] Saved radial comparison to {path_rad_cmp}")

    # Structure factors
    K_spinor = compute_kgrid(*density_spinor.shape)
    density_spinor_cp = cp.asarray(density_spinor, dtype=cp.float64)
    n_k_spinor_cp = cp.fft.fftn(density_spinor_cp)
    S_k_spinor = np.abs(cp.asnumpy(n_k_spinor_cp))**2
    k_centers, S_of_k_spinor = radial_structure_factor(S_k_spinor, K_spinor, n_bins=40)

    K_two = compute_kgrid(Lx, Ly, Lz)
    rho_two_cp = cp.asarray(rho_tot_3d, dtype=cp.float64)
    n_k_two_cp = cp.fft.fftn(rho_two_cp)
    S_k_two = np.abs(cp.asnumpy(n_k_two_cp))**2
    _, S_of_k_two = radial_structure_factor(S_k_two, K_two, n_bins=40)

    plt.figure(figsize=(6, 4))
    plt.plot(k_centers, S_of_k_spinor, marker="o", label="spinor emergent")
    plt.plot(k_centers, S_of_k_two, marker="s", label="two-fermion 3D")
    plt.xlabel("|k| (lattice units^-1)")
    plt.ylabel("S(k)")
    plt.title("Spinor vs two-fermion structure factors")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path_Sk_cmp = os.path.join(outdir, "spinor_vs_twofermion_Sk.png")
    plt.savefig(path_Sk_cmp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[twofermion] Saved S(k) comparison to {path_Sk_cmp}")


# =============================================================================
# Main
# =============================================================================

def main():
    global CUPY_IMPORT_ERROR

    parser = argparse.ArgumentParser(
        description="Spinor substrate evolution from noise with defrag & spin coupling (GPU)."
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
                        help="Strength of nonlocal defrag term (default -4.0; negative=attraction)")
    parser.add_argument("--sigma_defrag", type=float, default=3.0,
                        help="Width of Gaussian defrag kernel (default 3.0)")

    # Confinement
    parser.add_argument("--omega_conf", type=float, default=0.05,
                        help="Harmonic confining strength (default 0.05; 0 to disable)")

    # Spin coupling
    parser.add_argument("--gamma_spin", type=float, default=0.3,
                        help="Spin coupling strength gamma_spin (default 0.3)")

    # Output control
    parser.add_argument("--snapshot_every", type=int, default=500,
                        help="Save XY snapshots every N steps (default 500; 0=never)")

    # Comparison with substrate_engine_3d
    parser.add_argument("--compare_twofermion", action="store_true",
                        help="Also run substrate_engine_3d two-fermion model for comparison")
    parser.add_argument("--ref_L", type=int, default=2,
                        help="Lx=Ly=Lz for two-fermion reference (default 2)")

    args = parser.parse_args()

    if CUPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CuPy is required for this script.\n"
            f"Import error was: {CUPY_IMPORT_ERROR}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = ensure_outdir(script_dir, "outputs")

    # 1) Evolve spinor from noise
    density_tot, m_z, r_eff_history = evolve_spinor_from_noise_gpu(
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
        gamma_spin=args.gamma_spin,
        snapshot_every=args.snapshot_every,
        outdir=outdir,
    )

    # 2) Analyze emergent spinor structure
    analyze_spinor_final(density_tot, m_z, r_eff_history, outdir=outdir)

    # 3) Optional comparison with two-fermion model
    if args.compare_twofermion:
        compare_with_twofermion(density_tot, outdir=outdir, ref_L=args.ref_L)

    print("\n" + "=" * 70)
    print("SPINOR NOISE → EMERGENT STRUCTURE SIMULATION COMPLETE")
    print(f"Outputs in: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
