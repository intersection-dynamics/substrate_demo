from __future__ import annotations

"""
engines/substrate_engine.py

Unified substrate engine, v2 (with leapfrog integrator).

This module defines a toy "Hilbert substrate" on a 3D periodic lattice with:

- A bosonic sector: one or more complex scalar channels psi_B(x) ∈ ℂ^{N_B}
- A fermion-like sector: a multi-component complex field psi_F(x,color,spin)

It is NOT a faithful QFT, but a playground for:
- emergent "pointer states" (lumps, waves),
- defragmentation / clumping,
- basic diagnostics that experiments can build on.

Time evolution uses a leapfrog (symplectic-ish) integrator to improve
energy/norm behavior over the naive explicit scheme.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------


@dataclass
class SubstrateParams:
    # Lattice / time-stepping
    grid_size: int = 16
    dt: float = 0.01
    steps: int = 200

    # Bosonic sector: psi_B(x, b), b=0..N_B-1
    n_boson: int = 1
    # Boson mass^2 and quartic self-coupling lambda_B (len = n_boson or 1)
    boson_mass2: Tuple[float, ...] = (1.0,)
    boson_lambda4: Tuple[float, ...] = (1.0,)
    # Initial amplitude scale for boson fields (len = n_boson or 1)
    boson_init_amp: Tuple[float, ...] = (0.05,)

    # Fermionic-like sector: psi_F(x, color, spin)
    n_color: int = 1
    n_spin: int = 2
    # Fermion mass^2 and quartic self-coupling (per (color,spin) channel or broadcast)
    fermion_mass2: Tuple[float, ...] = (1.0,)
    fermion_lambda4: Tuple[float, ...] = (1.0,)
    # Initial amplitude scale for fermion-like fields
    fermion_init_amp: Tuple[float, ...] = (0.05,)

    # Simple boson–fermion cross-coupling:
    # V_int ∝ g_bf * (sum_b |psi_B|^2) * (sum_cs |psi_F|^2)
    g_bf: float = 0.0

    # Lump detection (on fermion density)
    lump_sigma_threshold: float = 2.0
    lump_min_voxels: int = 4

    # RNG
    seed: int = 1


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _laplacian_periodic(field: np.ndarray) -> np.ndarray:
    """
    3D discrete Laplacian with periodic boundary conditions.
    Works for both real and complex fields; last axes are treated as components.
    """
    xp = np.roll(field, 1, axis=0)
    xm = np.roll(field, -1, axis=0)
    yp = np.roll(field, 1, axis=1)
    ym = np.roll(field, -1, axis=1)
    zp = np.roll(field, 1, axis=2)
    zm = np.roll(field, -1, axis=2)
    return xp + xm + yp + ym + zp + zm - 6.0 * field


def _broadcast_params(arr: Tuple[float, ...], target_len: int, name: str) -> np.ndarray:
    arr_np = np.array(arr, dtype=np.float64)
    if arr_np.size == 1:
        arr_np = np.full((target_len,), float(arr_np[0]), dtype=np.float64)
    elif arr_np.size != target_len:
        raise ValueError(f"{name} must have length 1 or {target_len}, got {arr_np.size}")
    return arr_np


def _initial_state(params: SubstrateParams, rng: np.random.Generator):
    """
    Create initial conditions for psi_B, pi_B, psi_F, pi_F.

    psi_B: complex, shape (g,g,g,n_boson)
    pi_B:  complex "momentum" of same shape

    psi_F: complex, shape (g,g,g,n_color,n_spin)
    pi_F:  complex "momentum" of same shape
    """
    g = params.grid_size
    nb = params.n_boson
    nc = params.n_color
    ns = params.n_spin

    # Bosons
    psi_B = np.zeros((g, g, g, nb), dtype=np.complex128)
    pi_B = np.zeros_like(psi_B)
    bos_amp = _broadcast_params(params.boson_init_amp, nb, "boson_init_amp")
    noise_B = (
        rng.standard_normal((g, g, g, nb), dtype=np.float64)
        + 1j * rng.standard_normal((g, g, g, nb), dtype=np.float64)
    )
    psi_B += bos_amp.reshape((1, 1, 1, nb)) * noise_B

    # Fermion-like
    psi_F = np.zeros((g, g, g, nc, ns), dtype=np.complex128)
    pi_F = np.zeros_like(psi_F)
    n_fs = nc * ns
    ferm_amp_flat = _broadcast_params(params.fermion_init_amp, n_fs, "fermion_init_amp")
    ferm_amp = ferm_amp_flat.reshape((1, 1, 1, nc, ns))
    noise_F = (
        rng.standard_normal((g, g, g, nc, ns), dtype=np.float64)
        + 1j * rng.standard_normal((g, g, g, nc, ns), dtype=np.float64)
    )
    psi_F += ferm_amp * noise_F

    return psi_B, pi_B, psi_F, pi_F


def _energy_density(psi_B, pi_B, psi_F, pi_F, params: SubstrateParams) -> np.ndarray:
    """
    Compute local energy density (real scalar) on the lattice:

    For bosons:
      E_B = 0.5|pi_B|^2 + 0.5|grad psi_B|^2 + 0.5 m_B^2 |psi_B|^2 + 0.25 lambda_B |psi_B|^4

    For fermion-like:
      E_F = 0.5|pi_F|^2 + 0.5|grad psi_F|^2 + 0.5 m_F^2 |psi_F|^2 + 0.25 lambda_F |psi_F|^4

    Cross term:
      E_int = 0.5 * g_bf * (sum_b |psi_B|^2) * (sum_cs |psi_F|^2)
    """
    g = params.grid_size

    # Boson sector
    nb = params.n_boson
    bos_m2 = _broadcast_params(params.boson_mass2, nb, "boson_mass2")
    bos_l4 = _broadcast_params(params.boson_lambda4, nb, "boson_lambda4")

    # Gradients
    dpsiB_dx = 0.5 * (np.roll(psi_B, -1, axis=0) - np.roll(psi_B, 1, axis=0))
    dpsiB_dy = 0.5 * (np.roll(psi_B, -1, axis=1) - np.roll(psi_B, 1, axis=1))
    dpsiB_dz = 0.5 * (np.roll(psi_B, -1, axis=2) - np.roll(psi_B, 1, axis=2))
    gradB2 = np.abs(dpsiB_dx) ** 2 + np.abs(dpsiB_dy) ** 2 + np.abs(dpsiB_dz) ** 2  # (g,g,g,nb)
    modB2 = np.abs(psi_B) ** 2

    EB = (
        0.5 * np.abs(pi_B) ** 2
        + 0.5 * gradB2
        + 0.5 * bos_m2.reshape((1, 1, 1, nb)) * modB2
        + 0.25 * bos_l4.reshape((1, 1, 1, nb)) * (modB2 ** 2)
    ).sum(axis=-1)  # -> (g,g,g)

    # Fermion-like sector
    nc = params.n_color
    ns = params.n_spin
    n_fs = nc * ns
    ferm_m2_flat = _broadcast_params(params.fermion_mass2, n_fs, "fermion_mass2")
    ferm_l4_flat = _broadcast_params(params.fermion_lambda4, n_fs, "fermion_lambda4")
    ferm_m2 = ferm_m2_flat.reshape((1, 1, 1, nc, ns))
    ferm_l4 = ferm_l4_flat.reshape((1, 1, 1, nc, ns))

    dpsiF_dx = 0.5 * (np.roll(psi_F, -1, axis=0) - np.roll(psi_F, 1, axis=0))
    dpsiF_dy = 0.5 * (np.roll(psi_F, -1, axis=1) - np.roll(psi_F, 1, axis=1))
    dpsiF_dz = 0.5 * (np.roll(psi_F, -1, axis=2) - np.roll(psi_F, 1, axis=2))
    gradF2 = np.abs(dpsiF_dx) ** 2 + np.abs(dpsiF_dy) ** 2 + np.abs(dpsiF_dz) ** 2
    modF2 = np.abs(psi_F) ** 2

    EF = (
        0.5 * np.abs(pi_F) ** 2
        + 0.5 * gradF2
        + 0.5 * ferm_m2 * modF2
        + 0.25 * ferm_l4 * (modF2 ** 2)
    ).sum(axis=(-2, -1))  # -> (g,g,g)

    # Cross term
    rho_B = modB2.sum(axis=-1)  # (g,g,g)
    rho_F = modF2.sum(axis=(-2, -1))  # (g,g,g)
    E_int = 0.5 * float(params.g_bf) * rho_B * rho_F

    return EB + EF + E_int


def _compute_norm(psi_B, pi_B, psi_F, pi_F) -> float:
    """
    Simple norm-like diagnostic: mean over space of |psi|^2 + |pi|^2 for all components.
    """
    term_B = np.mean(np.abs(psi_B) ** 2 + np.abs(pi_B) ** 2)
    term_F = np.mean(np.abs(psi_F) ** 2 + np.abs(pi_F) ** 2)
    return float(term_B + term_F)


def _adaptive_lump_mask(
    rho: np.ndarray,
    sigma_threshold: float,
    min_voxels: int,
):
    """
    Adaptive thresholding on a scalar density rho(x):

      thresh = mean(rho) + sigma_threshold * std(rho)
      mask   = rho > thresh

    Lumps = connected components of mask using 6-neighbor connectivity
    with size >= min_voxels.

    Returns:
        mask (bool array), lump_count (int)
    """
    mu = float(rho.mean())
    sigma = float(rho.std())
    if sigma <= 0.0:
        return np.zeros_like(rho, dtype=bool), 0
    thresh = mu + sigma_threshold * sigma
    mask = rho > thresh
    if not mask.any():
        return mask, 0

    visited = np.zeros_like(mask, dtype=bool)
    g = rho.shape[0]
    lump_count = 0
    neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    xs, ys, zs = np.where(mask)
    for x0, y0, z0 in zip(xs, ys, zs):
        if visited[x0, y0, z0] or not mask[x0, y0, z0]:
            continue
        stack = [(x0, y0, z0)]
        visited[x0, y0, z0] = True
        size = 0
        while stack:
            x, y, z = stack.pop()
            size += 1
            for dx, dy, dz in neighbors:
                nx = (x + dx) % g
                ny = (y + dy) % g
                nz = (z + dz) % g
                if not visited[nx, ny, nz] and mask[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    stack.append((nx, ny, nz))
        if size >= min_voxels:
            lump_count += 1

    return mask, lump_count


def _com_periodic(rho: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Center-of-mass on a periodic lattice using circular means.

    rho: scalar density, shape (g,g,g)
    mask: optional boolean mask; if provided, restricts COM to masked region.

    Returns:
        np.array([x,y,z]) in lattice index units [0..g).
    """
    if mask is None:
        mask = np.ones_like(rho, dtype=bool)
    w = rho * mask
    total = w.sum()
    g = rho.shape[0]
    if total <= 0.0:
        return np.array([0.5 * (g - 1)] * 3, dtype=float)

    coords = []
    for axis in range(3):
        idx = np.arange(g)
        theta = 2.0 * np.pi * idx / g
        shape = [1, 1, 1]
        shape[axis] = g
        theta_grid = theta.reshape(shape)
        s = (w * np.sin(theta_grid)).sum()
        c = (w * np.cos(theta_grid)).sum()
        angle = np.arctan2(s, c)
        if angle < 0:
            angle += 2.0 * np.pi
        coord = angle * g / (2.0 * np.pi)
        coords.append(coord)
    return np.array(coords, dtype=float)


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------


def run_experiment(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evolve the unified substrate for a given number of steps with given parameters.

    raw_params: dict overriding SubstrateParams fields.

    Returns:
        {
          "params": {...},
          "metrics": {...},
          "diagnostics": {...},
          "verdicts": {...},
          "timeseries": {...},
        }
    """
    # Merge defaults with overrides
    base = SubstrateParams()
    for k, v in raw_params.items():
        if hasattr(base, k):
            setattr(base, k, v)
    params = base

    rng = _make_rng(params.seed)
    psi_B, pi_B, psi_F, pi_F = _initial_state(params, rng)

    # Initial diagnostics
    e = _energy_density(psi_B, pi_B, psi_F, pi_F, params)
    E_initial = float(e.mean())
    N_initial = _compute_norm(psi_B, pi_B, psi_F, pi_F)
    E_min = E_initial
    E_max = E_initial

    # Initial fermion density diagnostics
    rho_F0 = np.abs(psi_F) ** 2
    rho_F0 = rho_F0.sum(axis=(-2, -1))  # (g,g,g)
    ferm_mask0, ferm_lumps0 = _adaptive_lump_mask(
        rho_F0,
        params.lump_sigma_threshold,
        params.lump_min_voxels,
    )
    ferm_com0 = _com_periodic(rho_F0, ferm_mask0)

    psi_all0 = np.concatenate(
        [
            psi_B.reshape((*psi_B.shape[:3], -1)),
            psi_F.reshape((*psi_F.shape[:3], -1)),
        ],
        axis=-1,
    )
    phi_rms0 = float(np.sqrt(np.mean(np.abs(psi_all0) ** 2)))
    phi_max0 = float(np.max(np.abs(psi_all0)))

    # Time series buffers
    steps_trace: List[int] = [0]
    energy_trace: List[float] = [E_initial]
    norm_trace: List[float] = [N_initial]
    phi_rms_trace: List[float] = [phi_rms0]
    phi_max_trace: List[float] = [phi_max0]
    ferm_lump_trace: List[int] = [int(ferm_lumps0)]
    ferm_com_trace: List[List[float]] = [ferm_com0.tolist()]

    notes: List[str] = []

    sample_every = max(1, params.steps // 200)

    # Broadcast mass and coupling for dynamics
    nb = params.n_boson
    nc = params.n_color
    ns = params.n_spin
    n_fs = nc * ns

    bos_m2 = _broadcast_params(params.boson_mass2, nb, "boson_mass2").reshape(
        (1, 1, 1, nb)
    )
    bos_l4 = _broadcast_params(params.boson_lambda4, nb, "boson_lambda4").reshape(
        (1, 1, 1, nb)
    )
    ferm_m2 = _broadcast_params(params.fermion_mass2, n_fs, "fermion_mass2").reshape(
        (1, 1, 1, nc, ns)
    )
    ferm_l4 = _broadcast_params(params.fermion_lambda4, n_fs, "fermion_lambda4").reshape(
        (1, 1, 1, nc, ns)
    )

    g_bf = float(params.g_bf)

    # Helper to compute forces given current psi_B, psi_F
    def compute_forces(psi_B_local, psi_F_local):
        # Boson sector
        lapB = _laplacian_periodic(psi_B_local)
        modB2 = np.abs(psi_B_local) ** 2
        gradV_B = bos_m2 * psi_B_local + bos_l4 * modB2 * psi_B_local

        # Fermion-like sector
        lapF = _laplacian_periodic(psi_F_local)
        modF2 = np.abs(psi_F_local) ** 2
        gradV_F = ferm_m2 * psi_F_local + ferm_l4 * modF2 * psi_F_local

        # Cross-coupling
        rho_B_loc = modB2.sum(axis=-1, keepdims=True)          # (g,g,g,1)
        rho_F_loc = modF2.sum(axis=(-2, -1), keepdims=True)    # (g,g,g,1,1)

        # d/dpsi_B* of 0.5 g rho_B rho_F -> g psi_B rho_F
        gradV_B_cross = g_bf * psi_B_local * rho_F_loc[..., 0, 0]
        # d/dpsi_F* of 0.5 g rho_B rho_F -> g psi_F rho_B
        gradV_F_cross = g_bf * psi_F_local * rho_B_loc[..., 0, None]

        # Total forces (Klein-Gordon-like): lap - dV/dpsi*
        force_B_local = lapB - (gradV_B + gradV_B_cross)
        force_F_local = lapF - (gradV_F + gradV_F_cross)
        return force_B_local, force_F_local

    # ============================================================
    # Time evolution: leapfrog (symplectic-ish) integrator
    # ============================================================

    # Initial half-step for momenta
    force_B, force_F = compute_forces(psi_B, psi_F)
    pi_B = pi_B + 0.5 * params.dt * force_B
    pi_F = pi_F + 0.5 * params.dt * force_F

    for step in range(params.steps):
        # 1) Full step for psi using half-step momenta
        psi_B = psi_B + params.dt * pi_B
        psi_F = psi_F + params.dt * pi_F

        # 2) Compute forces at new psi
        force_B, force_F = compute_forces(psi_B, psi_F)

        # 3) Full step for momenta
        pi_B = pi_B + params.dt * force_B
        pi_F = pi_F + params.dt * force_F

        # 4) Diagnostics / sampling
        if (step + 1) % sample_every == 0 or step == params.steps - 1:
            e_step = _energy_density(psi_B, pi_B, psi_F, pi_F, params)
            e_mean = float(e_step.mean())
            E_min = min(E_min, e_mean)
            E_max = max(E_max, e_mean)

            N_step = _compute_norm(psi_B, pi_B, psi_F, pi_F)

            psi_all = np.concatenate(
                [
                    psi_B.reshape((*psi_B.shape[:3], -1)),
                    psi_F.reshape((*psi_F.shape[:3], -1)),
                ],
                axis=-1,
            )
            phi_rms = float(np.sqrt(np.mean(np.abs(psi_all) ** 2)))
            phi_max = float(np.max(np.abs(psi_all)))

            rho_F = np.abs(psi_F) ** 2
            rho_F = rho_F.sum(axis=(-2, -1))
            ferm_mask, ferm_lumps = _adaptive_lump_mask(
                rho_F,
                params.lump_sigma_threshold,
                params.lump_min_voxels,
            )
            ferm_com = _com_periodic(rho_F, ferm_mask)

            steps_trace.append(step + 1)
            energy_trace.append(e_mean)
            norm_trace.append(N_step)
            phi_rms_trace.append(phi_rms)
            phi_max_trace.append(phi_max)
            ferm_lump_trace.append(int(ferm_lumps))
            ferm_com_trace.append(ferm_com.tolist())

    # Final metrics
    e_final = _energy_density(psi_B, pi_B, psi_F, pi_F, params)
    E_final = float(e_final.mean())
    N_final = _compute_norm(psi_B, pi_B, psi_F, pi_F)

    abs_energy_drift = abs(E_final - E_initial)
    rel_energy_drift = abs_energy_drift / max(1e-12, abs(E_initial))
    abs_norm_drift = abs(N_final - N_initial)
    rel_norm_drift = abs_norm_drift / max(1e-12, abs(N_initial))

    abs_energy_tol = 5e-4
    rel_energy_tol = 0.5
    abs_norm_tol = 1e-3
    rel_norm_tol = 0.5

    energy_ok = (abs_energy_drift < abs_energy_tol) or (rel_energy_drift < rel_energy_tol)
    norm_ok = (abs_norm_drift < abs_norm_tol) or (rel_norm_drift < rel_norm_tol)
    stable = energy_ok and norm_ok

    if not energy_ok:
        notes.append(f"Energy drift abs={abs_energy_drift:.3e}, rel={rel_energy_drift:.3f}")
    if not norm_ok:
        notes.append(f"Norm drift abs={abs_norm_drift:.3e}, rel={rel_norm_drift:.3f}")

    rho_F_final = np.abs(psi_F) ** 2
    rho_F_final = rho_F_final.sum(axis=(-2, -1))
    ferm_mask_final, ferm_lumps_final = _adaptive_lump_mask(
        rho_F_final,
        params.lump_sigma_threshold,
        params.lump_min_voxels,
    )
    ferm_com_final = _com_periodic(rho_F_final, ferm_mask_final)

    results: Dict[str, Any] = {
        "params": asdict(params),
        "metrics": {
            "E_initial": E_initial,
            "E_final": E_final,
            "E_min": E_min,
            "E_max": E_max,
            "N_initial": N_initial,
            "N_final": N_final,
            "phi_rms_final": float(phi_rms_trace[-1]),
            "phi_max_final": float(phi_max_trace[-1]),
            "fermion_lump_count_final": int(ferm_lumps_final),
            "fermion_com_final": ferm_com_final.tolist(),
        },
        "diagnostics": {
            "abs_energy_drift": abs_energy_drift,
            "rel_energy_drift": rel_energy_drift,
            "abs_norm_drift": abs_norm_drift,
            "rel_norm_drift": rel_norm_drift,
            "stable": stable,
            "notes": notes,
        },
        "verdicts": {
            "has_fermion_lumps": int(ferm_lumps_final) > 0,
            "energy_reasonable": energy_ok,
            "norm_reasonable": norm_ok,
        },
        "timeseries": {
            "step": steps_trace,
            "energy": energy_trace,
            "norm": norm_trace,
            "phi_rms": phi_rms_trace,
            "phi_max": phi_max_trace,
            "fermion_lump_count": ferm_lump_trace,
            "fermion_com": ferm_com_trace,
        },
    }

    return results
