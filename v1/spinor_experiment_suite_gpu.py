#!/usr/bin/env python3
"""
spinor_experiment_suite_gpu.py

Test harness for the spinor substrate model:

  1) 10 random initial conditions:
       - final total energy
       - final r_eff
       - spin-texture similarity

  2) Rotation test (one converged state):
       - rotate by 45°, 90°, 180° in x–y
       - re-evolve
       - compare final energies and spin textures

  3) Parameter sweeps:
       - g_defrag:   -6 .. -2
       - gamma_spin: 0.1 .. 0.5
       - omega_conf: 0.01 .. 0.10
       - plot final r_eff vs parameter

  4) SU(2) rotation check:
       - apply SU(2) rotations to a converged state
       - verify global spin vector transforms correctly

This script is self-contained and uses CuPy on the GPU. It does NOT depend
on substrate_spinor_noise_emergent_gpu.py, although the physics is similar.
"""

import argparse
import os
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except ImportError as e:
    cp = None
    CUPY_IMPORT_ERROR = e
else:
    CUPY_IMPORT_ERROR = None

# Rotation for spatial grids
try:
    from scipy.ndimage import rotate as nd_rotate
except Exception:
    nd_rotate = None


# =============================================================================
# Utility: filesystem, grids, radial profiles, structure factors
# =============================================================================

def ensure_outdir(base_dir: str, subdir: str = "outputs") -> str:
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def compute_r_eff(density: np.ndarray) -> float:
    """
    RMS radius relative to center-of-mass, with lattice coordinates in indices.
    density: shape (Nx,Ny,Nz)
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
# Spinor substrate model (GPU)
# =============================================================================

@dataclass
class SpinorParams:
    Nx: int = 96
    Ny: int = 96
    Nz: int = 96
    dt: float = 0.01
    n_steps: int = 5000
    n_steps_refine: int = 2000
    m_eff: float = 1.0
    alpha: float = 0.2
    beta: float = 1.0
    g_defrag: float = -4.0
    sigma_defrag: float = 3.0
    omega_conf: float = 0.05
    gamma_spin: float = 0.3


def build_coordinate_grid_gpu(
    Nx: int,
    Ny: int,
    Nz: int
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    x = cp.arange(Nx) - 0.5 * (Nx - 1)
    y = cp.arange(Ny) - 0.5 * (Ny - 1)
    z = cp.arange(Nz) - 0.5 * (Nz - 1)
    X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
    R = cp.sqrt(X**2 + Y**2 + Z**2)
    return X, Y, Z, R


def build_conf_potential_gpu(R: cp.ndarray, omega_conf: float) -> cp.ndarray:
    if omega_conf <= 0:
        return cp.zeros_like(R)
    return 0.5 * (omega_conf ** 2) * (R ** 2)


def build_kinetic_factor_gpu(
    Nx: int,
    Ny: int,
    Nz: int,
    dt: float,
    m_eff: float
) -> cp.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    T_k = 0.5 * K2 / m_eff
    return cp.asarray(np.exp(-dt * T_k), dtype=cp.complex128)


def build_gaussian_kernel_kspace_gpu(
    Nx: int,
    Ny: int,
    Nz: int,
    sigma: float
) -> cp.ndarray:
    x = np.arange(Nx) - 0.5 * (Nx - 1)
    y = np.arange(Ny) - 0.5 * (Ny - 1)
    z = np.arange(Nz) - 0.5 * (Nz - 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R2 = X**2 + Y**2 + Z**2

    G = np.exp(-0.5 * R2 / (sigma ** 2))
    G /= G.sum()

    G_cp = cp.asarray(G, dtype=cp.float64)
    return cp.fft.fftn(G_cp)


def grad_sq_gpu(psi_s: cp.ndarray) -> cp.ndarray:
    """
    |∇psi_s|^2 using central finite differences with periodic boundaries.
    psi_s: shape (Nx,Ny,Nz)
    """
    dxp = cp.roll(psi_s, -1, axis=0) - psi_s
    dxm = psi_s - cp.roll(psi_s, 1, axis=0)
    dyp = cp.roll(psi_s, -1, axis=1) - psi_s
    dym = psi_s - cp.roll(psi_s, 1, axis=1)
    dzp = cp.roll(psi_s, -1, axis=2) - psi_s
    dzm = psi_s - cp.roll(psi_s, 1, axis=2)

    # symmetric derivative squared
    grad2 = (cp.abs(dxp + dxm) ** 2 +
             cp.abs(dyp + dym) ** 2 +
             cp.abs(dzp + dzm) ** 2) / 4.0
    return grad2


def compute_energy_gpu(
    psi: cp.ndarray,
    params: SpinorParams,
    G_k_cp: cp.ndarray,
    V_conf_cp: cp.ndarray,
) -> float:
    """
    Crude total "energy" functional for diagnostics.
    """
    Nx, Ny, Nz = psi.shape[1:]
    volume = Nx * Ny * Nz

    rho_up = cp.abs(psi[0]) ** 2
    rho_dn = cp.abs(psi[1]) ** 2
    rho_tot = rho_up + rho_dn
    m_z = rho_up - rho_dn

    # Kinetic energy
    grad2_up = grad_sq_gpu(psi[0])
    grad2_dn = grad_sq_gpu(psi[1])
    E_kin = cp.sum((grad2_up + grad2_dn) / (2.0 * params.m_eff)) / volume

    # Nonlocal convolution term
    rho_k = cp.fft.fftn(rho_tot)
    conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real

    # Local + nonlocal + conf + spin (all scalar densities)
    E_loc = cp.sum(params.alpha * rho_tot) / volume
    E_nl = cp.sum(0.5 * params.g_defrag * rho_tot * conv_cp) / volume
    E_quartic = cp.sum(0.5 * params.beta * rho_tot ** 2) / volume
    E_conf = cp.sum(V_conf_cp * rho_tot) / volume
    # simple spin energy: gamma_spin * m_z^2
    E_spin = cp.sum(0.5 * params.gamma_spin * m_z ** 2) / volume

    E_tot = (E_kin + E_loc + E_nl + E_quartic + E_conf + E_spin).real
    return float(E_tot)


def evolve_spinor_gpu(
    params: SpinorParams,
    outdir: str,
    seed: Optional[int] = None,
    psi_init: Optional[np.ndarray] = None,
    steps_override: Optional[int] = None,
    verbose_prefix: str = ""
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Imaginary-time evolution of the two-component spinor.

    If psi_init is None, start from random noise using 'seed'.
    Returns:
        density_tot_np, m_z_np, final_energy, final_r_eff
    """
    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    dt = params.dt
    n_steps = steps_override if steps_override is not None else params.n_steps

    print(f"{verbose_prefix}Spinor evolution on GPU: "
          f"{Nx}x{Ny}x{Nz}, n_steps={n_steps}")

    X, Y, Z, R = build_coordinate_grid_gpu(Nx, Ny, Nz)
    V_conf_cp = build_conf_potential_gpu(R, params.omega_conf)
    T_factor_cp = build_kinetic_factor_gpu(Nx, Ny, Nz, dt, params.m_eff)
    G_k_cp = build_gaussian_kernel_kspace_gpu(
        Nx, Ny, Nz, params.sigma_defrag
    )

    # Initial spinor psi
    if psi_init is None:
        rng = cp.random.default_rng(seed)
        psi = cp.empty((2, Nx, Ny, Nz), dtype=cp.complex128)
        envelope = cp.exp(-0.5 * (R / (0.5 * Nx)) ** 2)
        for s in range(2):
            psi_s = (rng.standard_normal((Nx, Ny, Nz)) +
                     1j * rng.standard_normal((Nx, Ny, Nz)))
            psi[s] = psi_s * envelope
    else:
        # psi_init is NumPy array on CPU, copy to GPU
        psi = cp.asarray(psi_init, dtype=cp.complex128)

    # Normalize
    norm = cp.sqrt(cp.sum(cp.abs(psi) ** 2))
    psi /= norm

    # Evolution loop
    for step in range(1, n_steps + 1):
        rho_up = cp.abs(psi[0]) ** 2
        rho_dn = cp.abs(psi[1]) ** 2
        rho_tot = rho_up + rho_dn
        m_z = rho_up - rho_dn

        rho_k = cp.fft.fftn(rho_tot)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real

        V0 = (params.alpha +
              params.beta * rho_tot +
              params.g_defrag * conv_cp +
              V_conf_cp)

        V_up = V0 + params.gamma_spin * m_z
        V_dn = V0 - params.gamma_spin * m_z

        # half-step potential
        psi[0] *= cp.exp(-0.5 * dt * V_up)
        psi[1] *= cp.exp(-0.5 * dt * V_dn)

        # kinetic
        psi_k = cp.fft.fftn(psi, axes=(1, 2, 3))
        psi_k *= T_factor_cp[None, :, :, :]
        psi = cp.fft.ifftn(psi_k, axes=(1, 2, 3))

        # second half-step
        rho_up = cp.abs(psi[0]) ** 2
        rho_dn = cp.abs(psi[1]) ** 2
        rho_tot = rho_up + rho_dn
        m_z = rho_up - rho_dn

        rho_k = cp.fft.fftn(rho_tot)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real
        V0 = (params.alpha +
              params.beta * rho_tot +
              params.g_defrag * conv_cp +
              V_conf_cp)

        V_up = V0 + params.gamma_spin * m_z
        V_dn = V0 - params.gamma_spin * m_z

        psi[0] *= cp.exp(-0.5 * dt * V_up)
        psi[1] *= cp.exp(-0.5 * dt * V_dn)

        # renormalize
        norm = cp.sqrt(cp.sum(cp.abs(psi) ** 2))
        psi /= norm

        if step % max(1, n_steps // 10) == 0 or step == 1:
            psi_np = cp.asnumpy(psi)
            density_tot_np = (np.abs(psi_np[0]) ** 2 +
                              np.abs(psi_np[1]) ** 2)
            density_tot_np /= density_tot_np.sum()
            r_eff = compute_r_eff(density_tot_np)
            print(f"{verbose_prefix}  step {step:6d}/{n_steps} "
                  f"r_eff ≈ {r_eff:.4f}")

    # Final diagnostics
    psi_np = cp.asnumpy(psi)
    rho_up_np = np.abs(psi_np[0]) ** 2
    rho_dn_np = np.abs(psi_np[1]) ** 2
    density_tot_np = rho_up_np + rho_dn_np
    total = density_tot_np.sum()
    if total > 0:
        density_tot_np /= total
    m_z_np = rho_up_np - rho_dn_np

    # Final energy
    psi_cp_final = cp.asarray(psi_np, dtype=cp.complex128)
    E_tot = compute_energy_gpu(psi_cp_final, params, G_k_cp, V_conf_cp)
    r_eff_final = compute_r_eff(density_tot_np)

    return density_tot_np, m_z_np, E_tot, r_eff_final


# =============================================================================
# Experiments
# =============================================================================

def spin_texture_similarity(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    Cosine similarity between flattened spin textures m1 and m2.
    """
    v1 = m1.flatten()
    v2 = m2.flatten()
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def experiment_random_ics(params: SpinorParams, outdir: str):
    """
    10 random initial conditions:
        - final energy spread
        - final r_eff spread
        - spin texture similarity matrix
    """
    print("\n================  RANDOM INITIAL CONDITIONS TEST  ================")
    n_runs = 10
    energies: List[float] = []
    r_effs: List[float] = []
    spin_textures: List[np.ndarray] = []

    for i in range(n_runs):
        seed = 1000 + i
        prefix = f"[run {i}] "
        density, m_z, E, r_eff = evolve_spinor_gpu(
            params, outdir=outdir, seed=seed, psi_init=None,
            steps_override=None, verbose_prefix=prefix
        )
        energies.append(E)
        r_effs.append(r_eff)
        spin_textures.append(m_z)

    energies = np.array(energies)
    r_effs = np.array(r_effs)

    print("\nRandom IC results:")
    print(f"  Energy: mean={energies.mean():.6f}, std={energies.std():.6f}")
    print(f"  r_eff : mean={r_effs.mean():.6f}, std={r_effs.std():.6f}")

    # Spin similarity
    sim_mat = np.zeros((n_runs, n_runs), dtype=float)
    for i in range(n_runs):
        for j in range(n_runs):
            sim_mat[i, j] = spin_texture_similarity(
                spin_textures[i], spin_textures[j]
            )

    # Save similarity matrix heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_mat, origin="lower", vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="spin texture cosine similarity")
    plt.xlabel("run index")
    plt.ylabel("run index")
    plt.title("Spin texture similarity (10 random ICs)")
    path_sim = os.path.join(outdir, "spinor_randomICs_spin_similarity.png")
    plt.savefig(path_sim, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[random] Spin similarity matrix saved to {path_sim}")

    # Histograms
    plt.figure(figsize=(6, 4))
    plt.hist(energies, bins=8)
    plt.xlabel("final energy")
    plt.ylabel("count")
    plt.title("Random ICs: final energy distribution")
    path_Ehist = os.path.join(outdir, "spinor_randomICs_energy_hist.png")
    plt.savefig(path_Ehist, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(r_effs, bins=8)
    plt.xlabel("final r_eff")
    plt.ylabel("count")
    plt.title("Random ICs: final r_eff distribution")
    path_rhist = os.path.join(outdir, "spinor_randomICs_reff_hist.png")
    plt.savefig(path_rhist, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[random] Summary plots saved to {path_Ehist} and {path_rhist}")


def rotate_spinor_xy(
    psi: np.ndarray,
    angle_deg: float
) -> np.ndarray:
    """
    Rotate spinor field in the x–y plane by angle_deg using SciPy ndimage.rotate.
    psi: shape (2,Nx,Ny,Nz) on CPU.
    """
    if nd_rotate is None:
        raise RuntimeError("scipy.ndimage.rotate not available.")
    psi_rot = np.empty_like(psi)
    for s in range(2):
        real = np.asanyarray(psi[s].real)
        imag = np.asanyarray(psi[s].imag)
        real_r = nd_rotate(real, angle=angle_deg, axes=(0, 1),
                           reshape=False, order=3, mode="nearest")
        imag_r = nd_rotate(imag, angle=angle_deg, axes=(0, 1),
                           reshape=False, order=3, mode="nearest")
        psi_rot[s] = real_r + 1j * imag_r
    return psi_rot


def apply_su2_rotation_z(psi: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Apply SU(2) spin rotation for rotation by 'angle_deg' about the z axis:

      U = exp(-i θ σ_z / 2) = diag(e^{-i θ/2}, e^{+i θ/2})

    psi: shape (2,Nx,Ny,Nz) on CPU.
    """
    theta = math.radians(angle_deg)
    phase_up = np.exp(-0.5j * theta)
    phase_dn = np.exp(+0.5j * theta)
    psi_rot = np.empty_like(psi)
    psi_rot[0] = phase_up * psi[0]
    psi_rot[1] = phase_dn * psi[1]
    return psi_rot


def experiment_rotation_test(params: SpinorParams, outdir: str):
    """
    Rotation test:
      - get one converged state
      - rotate by 45°, 90°, 180°
      - optionally apply SU(2) spin rotation
      - re-evolve
      - compare energies and spin textures
    """
    print("\n====================  ROTATION TEST  ====================")
    # 1) base run
    base_density, base_mz, base_E, base_reff = evolve_spinor_gpu(
        params, outdir=outdir, seed=1234, psi_init=None,
        steps_override=params.n_steps, verbose_prefix="[base] "
    )
    print(f"[base] E_final={base_E:.6f}, r_eff={base_reff:.4f}")

    # reconstruct spinor from base evolution by re-running with same seed
    # but capturing psi; hack: evolve once more with fewer steps and seed=1234
    # Instead, for this harness, we treat density/m_z only and re-use them as
    # diagnostic reference for similarity.
    base_mz_ref = base_mz

    angles = [45.0, 90.0, 180.0]
    results = []

    # To get psi_final_cpu we rerun with same seed but intercept at end
    # by modifying evolve_spinor_gpu slightly here:
    # (we'll just run a fresh evolution and keep psi this time)
    print("[rotation] Rerunning base with psi output for rotation...")
    # quick rerun with same parameters but we’ll intercept psi inside
    # We do it inline to avoid rewriting evolve_spinor_gpu:
    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    X, Y, Z, R = build_coordinate_grid_gpu(Nx, Ny, Nz)
    V_conf_cp = build_conf_potential_gpu(R, params.omega_conf)
    T_factor_cp = build_kinetic_factor_gpu(Nx, Ny, Nz, params.dt, params.m_eff)
    G_k_cp = build_gaussian_kernel_kspace_gpu(
        Nx, Ny, Nz, params.sigma_defrag
    )
    rng = cp.random.default_rng(1234)
    psi = cp.empty((2, Nx, Ny, Nz), dtype=cp.complex128)
    envelope = cp.exp(-0.5 * (R / (0.5 * Nx)) ** 2)
    for s in range(2):
        psi_s = (rng.standard_normal((Nx, Ny, Nz)) +
                 1j * rng.standard_normal((Nx, Ny, Nz)))
        psi[s] = psi_s * envelope
    norm = cp.sqrt(cp.sum(cp.abs(psi) ** 2))
    psi /= norm
    for step in range(1, params.n_steps + 1):
        rho_up = cp.abs(psi[0]) ** 2
        rho_dn = cp.abs(psi[1]) ** 2
        rho_tot = rho_up + rho_dn
        m_z = rho_up - rho_dn
        rho_k = cp.fft.fftn(rho_tot)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real
        V0 = (params.alpha +
              params.beta * rho_tot +
              params.g_defrag * conv_cp +
              V_conf_cp)
        V_up = V0 + params.gamma_spin * m_z
        V_dn = V0 - params.gamma_spin * m_z
        psi[0] *= cp.exp(-0.5 * params.dt * V_up)
        psi[1] *= cp.exp(-0.5 * params.dt * V_dn)
        psi_k = cp.fft.fftn(psi, axes=(1, 2, 3))
        psi_k *= T_factor_cp[None, :, :, :]
        psi = cp.fft.ifftn(psi_k, axes=(1, 2, 3))
        rho_up = cp.abs(psi[0]) ** 2
        rho_dn = cp.abs(psi[1]) ** 2
        rho_tot = rho_up + rho_dn
        m_z = rho_up - rho_dn
        rho_k = cp.fft.fftn(rho_tot)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real
        V0 = (params.alpha +
              params.beta * rho_tot +
              params.g_defrag * conv_cp +
              V_conf_cp)
        V_up = V0 + params.gamma_spin * m_z
        V_dn = V0 - params.gamma_spin * m_z
        psi[0] *= cp.exp(-0.5 * params.dt * V_up)
        psi[1] *= cp.exp(-0.5 * params.dt * V_dn)
        norm = cp.sqrt(cp.sum(cp.abs(psi) ** 2))
        psi /= norm
    psi_base_cpu = cp.asnumpy(psi)

    for angle in angles:
        print(f"\n[rotation] Angle = {angle} degrees")
        psi_rot_cpu = rotate_spinor_xy(psi_base_cpu, angle)
        # optional SU(2) spin rotation about z
        psi_rot_cpu = apply_su2_rotation_z(psi_rot_cpu, angle)

        density_rot, m_z_rot, E_rot, r_eff_rot = evolve_spinor_gpu(
            params, outdir=outdir, seed=None,
            psi_init=psi_rot_cpu,
            steps_override=params.n_steps_refine,
            verbose_prefix=f"[rot {angle:5.1f}°] "
        )

        sim_spin = spin_texture_similarity(base_mz_ref, m_z_rot)
        print(f"[rot {angle:5.1f}°] E_final={E_rot:.6f}, "
              f"r_eff={r_eff_rot:.4f}, "
              f"spin_similarity_to_base={sim_spin:.4f}")

        results.append((angle, E_rot, r_eff_rot, sim_spin))

    # Plot E and similarity vs angle
    angles_arr = np.array([r[0] for r in results])
    E_arr = np.array([r[1] for r in results])
    reff_arr = np.array([r[2] for r in results])
    sim_arr = np.array([r[3] for r in results])

    plt.figure(figsize=(6, 4))
    plt.plot(angles_arr, E_arr, marker="o")
    plt.xlabel("rotation angle (deg)")
    plt.ylabel("final energy")
    plt.title("Rotation test: final energy vs angle")
    plt.grid(True, alpha=0.3)
    path_E = os.path.join(outdir, "spinor_rotation_energy_vs_angle.png")
    plt.savefig(path_E, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(angles_arr, sim_arr, marker="o")
    plt.xlabel("rotation angle (deg)")
    plt.ylabel("spin texture similarity to base")
    plt.title("Rotation test: spin similarity vs angle")
    plt.grid(True, alpha=0.3)
    path_S = os.path.join(outdir, "spinor_rotation_similarity_vs_angle.png")
    plt.savefig(path_S, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[rotation] Summary plots saved to {path_E} and {path_S}")


def experiment_param_sweep(params: SpinorParams, outdir: str):
    """
    Parameter sweeps for:
        g_defrag    : -6 .. -2
        gamma_spin  : 0.1 .. 0.5
        omega_conf  : 0.01 .. 0.10
    Plot final r_eff vs parameter.
    """
    print("\n====================  PARAMETER SWEEPS  ====================")

    # 1) g_defrag sweep
    g_vals = [-6.0, -5.0, -4.0, -3.0, -2.0]
    r_g = []
    for g in g_vals:
        p = SpinorParams(**asdict(params))
        p.g_defrag = g
        print(f"\n[sweep g_defrag] g={g}")
        _, _, _, r_eff = evolve_spinor_gpu(
            p, outdir=outdir, seed=2000 + int(10*g),
            psi_init=None, steps_override=p.n_steps,
            verbose_prefix=f"[g={g:4.1f}] "
        )
        r_g.append(r_eff)
    r_g = np.array(r_g)
    plt.figure(figsize=(6, 4))
    plt.plot(g_vals, r_g, marker="o")
    plt.xlabel("g_defrag")
    plt.ylabel("final r_eff")
    plt.title("Parameter sweep: r_eff vs g_defrag")
    plt.grid(True, alpha=0.3)
    path_g = os.path.join(outdir, "spinor_sweep_reff_vs_gdefrag.png")
    plt.savefig(path_g, dpi=150, bbox_inches="tight")
    plt.close()

    # 2) gamma_spin sweep
    gs_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    r_gamma = []
    for gs in gs_vals:
        p = SpinorParams(**asdict(params))
        p.gamma_spin = gs
        print(f"\n[sweep gamma_spin] gamma={gs}")
        _, _, _, r_eff = evolve_spinor_gpu(
            p, outdir=outdir, seed=3000 + int(100*gs),
            psi_init=None, steps_override=p.n_steps,
            verbose_prefix=f"[gamma={gs:3.1f}] "
        )
        r_gamma.append(r_eff)
    r_gamma = np.array(r_gamma)
    plt.figure(figsize=(6, 4))
    plt.plot(gs_vals, r_gamma, marker="o")
    plt.xlabel("gamma_spin")
    plt.ylabel("final r_eff")
    plt.title("Parameter sweep: r_eff vs gamma_spin")
    plt.grid(True, alpha=0.3)
    path_gamma = os.path.join(outdir, "spinor_sweep_reff_vs_gamma_spin.png")
    plt.savefig(path_gamma, dpi=150, bbox_inches="tight")
    plt.close()

    # 3) omega_conf sweep
    om_vals = [0.01, 0.03, 0.05, 0.07, 0.10]
    r_om = []
    for om in om_vals:
        p = SpinorParams(**asdict(params))
        p.omega_conf = om
        print(f"\n[sweep omega_conf] omega={om}")
        _, _, _, r_eff = evolve_spinor_gpu(
            p, outdir=outdir, seed=4000 + int(100*om),
            psi_init=None, steps_override=p.n_steps,
            verbose_prefix=f"[omega={om:4.2f}] "
        )
        r_om.append(r_eff)
    r_om = np.array(r_om)
    plt.figure(figsize=(6, 4))
    plt.plot(om_vals, r_om, marker="o")
    plt.xlabel("omega_conf")
    plt.ylabel("final r_eff")
    plt.title("Parameter sweep: r_eff vs omega_conf")
    plt.grid(True, alpha=0.3)
    path_om = os.path.join(outdir, "spinor_sweep_reff_vs_omega_conf.png")
    plt.savefig(path_om, dpi=150, bbox_inches="tight")
    plt.close()

    print("[sweep] Plots saved:")
    print(" ", path_g)
    print(" ", path_gamma)
    print(" ", path_om)


def global_spin_vector(psi: np.ndarray) -> np.ndarray:
    """
    Global spin expectation values S_x, S_y, S_z for a spinor field psi.
    psi: shape (2,Nx,Ny,Nz), CPU.
    """
    up = psi[0].reshape(-1)
    dn = psi[1].reshape(-1)
    # S_x = ψ† σ_x ψ = 2 Re(up* conj(dn))
    # S_y = ψ† σ_y ψ = 2 Im(up* conj(dn))
    # S_z = |up|^2 - |dn|^2
    Sx = 2.0 * np.real(np.vdot(up, dn))      # conj(up)*dn
    Sy = 2.0 * np.imag(np.vdot(up, dn))      # conj(up)*dn (fixed)
    Sz = float((np.abs(up) ** 2 - np.abs(dn) ** 2).sum())
    return np.array([Sx, Sy, Sz], dtype=float)


def su2_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """
    SU(2) rotation matrix for rotation by angle_deg about axis x,y,z.
    """
    theta = math.radians(angle_deg)
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    if axis == "z":
        # exp(-i θ σ_z /2)
        return np.array([[c - 1j * s, 0],
                         [0, c + 1j * s]], dtype=complex)
    elif axis == "x":
        # exp(-i θ σ_x /2) = cos θ/2 I - i sin θ/2 σ_x
        return np.array([[c, -1j * s],
                         [-1j * s, c]], dtype=complex)
    elif axis == "y":
        # exp(-i θ σ_y /2)
        return np.array([[c, -s],
                         [s, c]], dtype=complex)
    else:
        raise ValueError("axis must be 'x','y', or 'z'")


def so3_rotation(axis: str, angle_deg: float, vec: np.ndarray) -> np.ndarray:
    """
    Apply SO(3) rotation to 3-vector vec for comparison with SU(2).
    """
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    if axis == "z":
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=float)
    elif axis == "x":
        R = np.array([[1, 0,  0],
                      [0, c, -s],
                      [0, s,  c]], dtype=float)
    elif axis == "y":
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=float)
    else:
        raise ValueError("axis must be 'x','y', or 'z'")
    return R @ vec


def apply_su2_to_field(psi: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Apply 2x2 SU(2) matrix U to spinor field psi at every lattice site.
    psi: (2,Nx,Ny,Nz)
    """
    up = psi[0]
    dn = psi[1]
    psi_new0 = U[0, 0] * up + U[0, 1] * dn
    psi_new1 = U[1, 0] * up + U[1, 1] * dn
    psi_new = np.empty_like(psi)
    psi_new[0] = psi_new0
    psi_new[1] = psi_new1
    return psi_new


def experiment_su2_check(params: SpinorParams, outdir: str):
    """
    SU(2) transformation check:
      - take a converged state
      - compute global spin vector S
      - apply SU(2) rotations about x,z
      - check that S' ≈ R S
    """
    print("\n====================  SU(2) CHECK  ====================")
    density, m_z, E, r_eff = evolve_spinor_gpu(
        params, outdir=outdir, seed=9876, psi_init=None,
        steps_override=params.n_steps, verbose_prefix="[su2 base] "
    )
    print(f"[su2 base] E_final={E:.6f}, r_eff={r_eff:.4f}")

    # Reconstruct psi similarly to rotation test
    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    X, Y, Z, R = build_coordinate_grid_gpu(Nx, Ny, Nz)
    V_conf_cp = build_conf_potential_gpu(R, params.omega_conf)
    T_factor_cp = build_kinetic_factor_gpu(Nx, Ny, Nz, params.dt, params.m_eff)
    G_k_cp = build_gaussian_kernel_kspace_gpu(
        Nx, Ny, Nz, params.sigma_defrag
    )
    rng = cp.random.default_rng(9876)
    psi = cp.empty((2, Nx, Ny, Nz), dtype=cp.complex128)
    envelope = cp.exp(-0.5 * (R / (0.5 * Nx)) ** 2)
    for s in range(2):
        psi_s = (rng.standard_normal((Nx, Ny, Nz)) +
                 1j * rng.standard_normal((Nx, Ny, Nz)))
        psi[s] = psi_s * envelope
    norm = cp.sqrt(cp.sum(cp.abs(psi) ** 2))
    psi /= norm
    for step in range(1, params.n_steps + 1):
        rho_up = cp.abs(psi[0]) ** 2
        rho_dn = cp.abs(psi[1]) ** 2
        rho_tot = rho_up + rho_dn
        m_z = rho_up - rho_dn
        rho_k = cp.fft.fftn(rho_tot)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real
        V0 = (params.alpha +
              params.beta * rho_tot +
              params.g_defrag * conv_cp +
              V_conf_cp)
        V_up = V0 + params.gamma_spin * m_z
        V_dn = V0 - params.gamma_spin * m_z
        psi[0] *= cp.exp(-0.5 * params.dt * V_up)
        psi[1] *= cp.exp(-0.5 * params.dt * V_dn)
        psi_k = cp.fft.fftn(psi, axes=(1, 2, 3))
        psi_k *= T_factor_cp[None, :, :, :]
        psi = cp.fft.ifftn(psi_k, axes=(1, 2, 3))
        rho_up = cp.abs(psi[0]) ** 2
        rho_dn = cp.abs(psi[1]) ** 2
        rho_tot = rho_up + rho_dn
        m_z = rho_up - rho_dn
        rho_k = cp.fft.fftn(rho_tot)
        conv_cp = cp.fft.ifftn(rho_k * G_k_cp).real
        V0 = (params.alpha +
              params.beta * rho_tot +
              params.g_defrag * conv_cp +
              V_conf_cp)
        V_up = V0 + params.gamma_spin * m_z
        V_dn = V0 - params.gamma_spin * m_z
        psi[0] *= cp.exp(-0.5 * params.dt * V_up)
        psi[1] *= cp.exp(-0.5 * params.dt * V_dn)
        norm = cp.sqrt(cp.sum(cp.abs(psi) ** 2))
        psi /= norm
    psi_base = cp.asnumpy(psi)

    S_base = global_spin_vector(psi_base)
    print(f"[su2 base] Global spin S = {S_base}")

    axes = ["z", "x"]
    angles = [30.0, 60.0, 90.0]

    for axis in axes:
        for ang in angles:
            U = su2_matrix(axis, ang)
            psi_rot = apply_su2_to_field(psi_base, U)
            S_rot = global_spin_vector(psi_rot)
            S_expected = so3_rotation(axis, ang, S_base)
            diff = S_rot - S_expected
            err = np.linalg.norm(diff)
            print(f"[su2 {axis}, {ang:5.1f}°] "
                  f"S_rot={S_rot}, S_expected={S_expected}, "
                  f"||ΔS||={err:.3e}")


# =============================================================================
# Main
# =============================================================================

def main():
    global CUPY_IMPORT_ERROR
    if CUPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CuPy is required for this script.\n"
            f"Import error was: {CUPY_IMPORT_ERROR}"
        )

    parser = argparse.ArgumentParser(
        description="Spinor substrate experiment suite (GPU)."
    )

    # Lattice
    parser.add_argument("--Nx", type=int, default=96, help="grid size in x")
    parser.add_argument("--Ny", type=int, default=96, help="grid size in y")
    parser.add_argument("--Nz", type=int, default=96, help="grid size in z")

    # Imaginary time
    parser.add_argument("--dt", type=float, default=0.01, help="time step")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="relaxation steps for base runs")
    parser.add_argument("--n_steps_refine", type=int, default=2000,
                        help="steps for re-relaxation (rotation test)")
    parser.add_argument("--m_eff", type=float, default=1.0,
                        help="effective mass")

    # Potential parameters
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--g_defrag", type=float, default=-4.0)
    parser.add_argument("--sigma_defrag", type=float, default=3.0)
    parser.add_argument("--omega_conf", type=float, default=0.05)
    parser.add_argument("--gamma_spin", type=float, default=0.3)

    # What to run
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "random", "rotation", "sweep", "su2"],
        help="which experiment to run"
    )

    args = parser.parse_args()

    params = SpinorParams(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        dt=args.dt,
        n_steps=args.n_steps,
        n_steps_refine=args.n_steps_refine,
        m_eff=args.m_eff,
        alpha=args.alpha,
        beta=args.beta,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        omega_conf=args.omega_conf,
        gamma_spin=args.gamma_spin,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = ensure_outdir(script_dir, "outputs")

    print("Spinor experiment suite with params:")
    for k, v in asdict(params).items():
        print(f"  {k} = {v}")
    print(f"Outputs will go to: {outdir}")

    if args.mode in ("all", "random"):
        experiment_random_ics(params, outdir)

    if args.mode in ("all", "rotation"):
        experiment_rotation_test(params, outdir)

    if args.mode in ("all", "sweep"):
        experiment_param_sweep(params, outdir)

    if args.mode in ("all", "su2"):
        experiment_su2_check(params, outdir)

    print("\n====================  SUITE COMPLETE  ====================")


if __name__ == "__main__":
    main()
