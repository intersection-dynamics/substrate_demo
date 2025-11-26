from __future__ import annotations

"""
engines/substrate_engine.py

Unified substrate engine, v3

- Symplectic-ish leapfrog integrator for (psi, pi) evolution.
- Optional GPU acceleration via CuPy (use_gpu flag).
- Bosonic sector: psi_B(x, b), complex, b=0..N_B-1
- Fermion-like sector: psi_F(x, color, spin), complex.

This is a toy Hilbert-substrate playground for:
- emergent "pointer states" (fermion-like lumps, bosonic modes),
- defragmentation / clumping behavior,
- energy/norm diagnostics.

Usage from experiments:
    from engines import substrate_engine as engine
    results = engine.run_experiment({...})
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as _np

try:
    import cupy as _cp
except ImportError:  # CuPy not installed
    _cp = None


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------


@dataclass
class SubstrateParams:
    # Lattice / time-stepping
    grid_size: int = 16
    dt: float = 0.01
    steps: int = 200

    # Bosonic sector: psi_B(x, b), b=0..N_B-1
    n_boson: int = 1
    boson_mass2: Tuple[float, ...] = (1.0,)
    boson_lambda4: Tuple[float, ...] = (1.0,)
    boson_init_amp: Tuple[float, ...] = (0.05,)

    # Fermion-like sector: psi_F(x, color, spin)
    n_color: int = 1
    n_spin: int = 2
    fermion_mass2: Tuple[float, ...] = (1.0,)
    fermion_lambda4: Tuple[float, ...] = (1.0,)
    fermion_init_amp: Tuple[float, ...] = (0.05,)

    # Boson–fermion cross-coupling:
    # V_int ∝ g_bf * (sum_b |psi_B|^2) * (sum_cs |psi_F|^2)
    g_bf: float = 0.0

    # Lump detection (on fermion density)
    lump_sigma_threshold: float = 2.0
    lump_min_voxels: int = 4

    # RNG
    seed: int = 1

    # Backend
    #   use_gpu = True  -> try CuPy; fall back to NumPy if unavailable
    #   use_gpu = False -> force NumPy
    use_gpu: bool = False


# ---------------------------------------------------------------------
# Backend helpers (NumPy vs CuPy)
# ---------------------------------------------------------------------


def _get_backend(use_gpu: bool):
    """
    Decide whether to use NumPy or CuPy.

    Returns:
        xp      : array module (numpy or cupy)
        backend : "numpy" or "cupy"
    """
    if use_gpu and _cp is not None:
        return _cp, "cupy"
    else:
        return _np, "numpy"


def _xp_array(obj, xp):
    return xp.array(obj)


def _broadcast_params_np(arr: Tuple[float, ...], target_len: int, name: str) -> _np.ndarray:
    arr_np = _np.array(arr, dtype=_np.float64)
    if arr_np.size == 1:
        arr_np = _np.full((target_len,), float(arr_np[0]), dtype=_np.float64)
    elif arr_np.size != target_len:
        raise ValueError(f"{name} must have length 1 or {target_len}, got {arr_np.size}")
    return arr_np


def _scalar(x, backend: str) -> float:
    """
    Convert a 0-d array / scalar from xp (numpy or cupy) into a Python float.
    """
    if backend == "cupy":
        return float(x.get())
    else:
        return float(x)


def _to_numpy(arr, backend: str):
    """
    Convert xp array to NumPy array (no-op if already NumPy).
    """
    if backend == "cupy":
        return _cp.asnumpy(arr)
    else:
        return _np.asarray(arr)


# ---------------------------------------------------------------------
# Core lattice helpers (xp-agnostic)
# ---------------------------------------------------------------------


def _laplacian_periodic(field, xp):
    """
    3D discrete Laplacian with periodic boundary conditions.
    Works for real/complex, last axes are treated as components.
    """
    xp_roll = xp.roll
    xp_field = field
    xp_xp = xp_roll(xp_field, 1, axis=0)
    xm = xp_roll(xp_field, -1, axis=0)
    yp = xp_roll(xp_field, 1, axis=1)
    ym = xp_roll(xp_field, -1, axis=1)
    zp = xp_roll(xp_field, 1, axis=2)
    zm = xp_roll(xp_field, -1, axis=2)
    return xp_xp + xm + yp + ym + zp + zm - 6.0 * xp_field


def _initial_state(params: SubstrateParams, xp, backend: str):
    """
    Create initial conditions for psi_B, pi_B, psi_F, pi_F using NumPy RNG,
    then cast to xp (NumPy or CuPy).
    """
    g = params.grid_size
    nb = params.n_boson
    nc = params.n_color
    ns = params.n_spin

    rng = _np.random.default_rng(params.seed)

    # Bosons
    psi_B_np = _np.zeros((g, g, g, nb), dtype=_np.complex128)
    pi_B_np = _np.zeros_like(psi_B_np)
    bos_amp_np = _broadcast_params_np(params.boson_init_amp, nb, "boson_init_amp")

    noise_B = (
        rng.standard_normal((g, g, g, nb), dtype=_np.float64)
        + 1j * rng.standard_normal((g, g, g, nb), dtype=_np.float64)
    )
    psi_B_np += bos_amp_np.reshape((1, 1, 1, nb)) * noise_B

    # Fermion-like
    psi_F_np = _np.zeros((g, g, g, nc, ns), dtype=_np.complex128)
    pi_F_np = _np.zeros_like(psi_F_np)
    n_fs = nc * ns
    ferm_amp_flat = _broadcast_params_np(params.fermion_init_amp, n_fs, "fermion_init_amp")
    ferm_amp_np = ferm_amp_flat.reshape((1, 1, 1, nc, ns))
    noise_F = (
        rng.standard_normal((g, g, g, nc, ns), dtype=_np.float64)
        + 1j * rng.standard_normal((g, g, g, nc, ns), dtype=_np.float64)
    )
    psi_F_np += ferm_amp_np * noise_F

    # Cast to xp
    psi_B = xp.array(psi_B_np)
    pi_B = xp.array(pi_B_np)
    psi_F = xp.array(psi_F_np)
    pi_F = xp.array(pi_F_np)

    return psi_B, pi_B, psi_F, pi_F


def _energy_density(psi_B, pi_B, psi_F, pi_F, params: SubstrateParams, xp, backend: str):
    """
    Local energy density on the lattice:

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
    bos_m2_np = _broadcast_params_np(params.boson_mass2, nb, "boson_mass2")
    bos_l4_np = _broadcast_params_np(params.boson_lambda4, nb, "boson_lambda4")
    bos_m2 = xp.array(bos_m2_np).reshape((1, 1, 1, nb))
    bos_l4 = xp.array(bos_l4_np).reshape((1, 1, 1, nb))

    # Gradients
    dpsiB_dx = 0.5 * (xp.roll(psi_B, -1, axis=0) - xp.roll(psi_B, 1, axis=0))
    dpsiB_dy = 0.5 * (xp.roll(psi_B, -1, axis=1) - xp.roll(psi_B, 1, axis=1))
    dpsiB_dz = 0.5 * (xp.roll(psi_B, -1, axis=2) - xp.roll(psi_B, 1, axis=2))
    gradB2 = xp.abs(dpsiB_dx) ** 2 + xp.abs(dpsiB_dy) ** 2 + xp.abs(dpsiB_dz) ** 2  # (g,g,g,nb)
    modB2 = xp.abs(psi_B) ** 2

    EB = (
        0.5 * xp.abs(pi_B) ** 2
        + 0.5 * gradB2
        + 0.5 * bos_m2 * modB2
        + 0.25 * bos_l4 * (modB2 ** 2)
    ).sum(axis=-1)  # -> (g,g,g)

    # Fermion-like sector
    nc = params.n_color
    ns = params.n_spin
    n_fs = nc * ns
    ferm_m2_np = _broadcast_params_np(params.fermion_mass2, n_fs, "fermion_mass2")
    ferm_l4_np = _broadcast_params_np(params.fermion_lambda4, n_fs, "fermion_lambda4")
    ferm_m2 = xp.array(ferm_m2_np).reshape((1, 1, 1, nc, ns))
    ferm_l4 = xp.array(ferm_l4_np).reshape((1, 1, 1, nc, ns))

    dpsiF_dx = 0.5 * (xp.roll(psi_F, -1, axis=0) - xp.roll(psi_F, 1, axis=0))
    dpsiF_dy = 0.5 * (xp.roll(psi_F, -1, axis=1) - xp.roll(psi_F, 1, axis=1))
    dpsiF_dz = 0.5 * (xp.roll(psi_F, -1, axis=2) - xp.roll(psi_F, 1, axis=2))
    gradF2 = xp.abs(dpsiF_dx) ** 2 + xp.abs(dpsiF_dy) ** 2 + xp.abs(dpsiF_dz) ** 2
    modF2 = xp.abs(psi_F) ** 2

    EF = (
        0.5 * xp.abs(pi_F) ** 2
        + 0.5 * gradF2
        + 0.5 * ferm_m2 * modF2
        + 0.25 * ferm_l4 * (modF2 ** 2)
    ).sum(axis=(-2, -1))  # -> (g,g,g)

    # Cross term
    rho_B = modB2.sum(axis=-1)  # (g,g,g)
    rho_F = modF2.sum(axis=(-2, -1))  # (g,g,g)
    E_int = 0.5 * float(params.g_bf) * rho_B * rho_F

    return EB + E_int + EF


def _compute_norm(psi_B, pi_B, psi_F, pi_F, xp, backend: str) -> float:
    term_B = xp.mean(xp.abs(psi_B) ** 2 + xp.abs(pi_B) ** 2)
    term_F = xp.mean(xp.abs(psi_F) ** 2 + xp.abs(pi_F) ** 2)
    return _scalar(term_B + term_F, backend)


def _adaptive_lump_mask(rho_cpu: _np.ndarray, sigma_threshold: float, min_voxels: int):
    """
    Adaptive thresholding on a scalar density rho(x) [NumPy]:

      thresh = mean(rho) + sigma_threshold * std(rho)
      mask   = rho > thresh

    Lumps = connected components of mask using 6-neighbor connectivity
    with size >= min_voxels.

    Returns:
        mask (bool NumPy array), lump_count (int)
    """
    mu = float(rho_cpu.mean())
    sigma = float(rho_cpu.std())
    if sigma <= 0.0:
        return _np.zeros_like(rho_cpu, dtype=bool), 0
    thresh = mu + sigma_threshold * sigma
    mask = rho_cpu > thresh
    if not mask.any():
        return mask, 0

    visited = _np.zeros_like(mask, dtype=bool)
    g = rho_cpu.shape[0]
    lump_count = 0
    neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    xs, ys, zs = _np.where(mask)
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


def _com_periodic(rho_cpu: _np.ndarray, mask: _np.ndarray | None = None) -> _np.ndarray:
    """
    Center-of-mass on a periodic lattice using circular means (NumPy).

    rho_cpu: scalar density, shape (g,g,g)
    mask: optional boolean mask.

    Returns:
        np.array([x,y,z]) in lattice index units [0..g).
    """
    if mask is None:
        mask = _np.ones_like(rho_cpu, dtype=bool)
    w = rho_cpu * mask
    total = w.sum()
    g = rho_cpu.shape[0]
    if total <= 0.0:
        return _np.array([0.5 * (g - 1)] * 3, dtype=float)

    coords = []
    for axis in range(3):
        idx = _np.arange(g)
        theta = 2.0 * _np.pi * idx / g
        shape = [1, 1, 1]
        shape[axis] = g
        theta_grid = theta.reshape(shape)
        s = (w * _np.sin(theta_grid)).sum()
        c = (w * _np.cos(theta_grid)).sum()
        angle = _np.arctan2(s, c)
        if angle < 0:
            angle += 2.0 * _np.pi
        coord = angle * g / (2.0 * _np.pi)
        coords.append(coord)
    return _np.array(coords, dtype=float)


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

    xp, backend = _get_backend(params.use_gpu)

    # Initial state
    psi_B, pi_B, psi_F, pi_F = _initial_state(params, xp, backend)

    # Initial diagnostics
    e0 = _energy_density(psi_B, pi_B, psi_F, pi_F, params, xp, backend)
    E_initial = _scalar(e0.mean(), backend)
    N_initial = _compute_norm(psi_B, pi_B, psi_F, pi_F, xp, backend)
    E_min = E_initial
    E_max = E_initial

    # Initial fermion density diagnostics (on CPU)
    rho_F0_xp = xp.abs(psi_F) ** 2
    rho_F0_xp = rho_F0_xp.sum(axis=(-2, -1))
    rho_F0 = _to_numpy(rho_F0_xp, backend)
    ferm_mask0, ferm_lumps0 = _adaptive_lump_mask(
        rho_F0,
        params.lump_sigma_threshold,
        params.lump_min_voxels,
    )
    ferm_com0 = _com_periodic(rho_F0, ferm_mask0)

    psi_all0_xp = xp.concatenate(
        [
            psi_B.reshape((*psi_B.shape[:3], -1)),
            psi_F.reshape((*psi_F.shape[:3], -1)),
        ],
        axis=-1,
    )
    phi_rms0 = _scalar(xp.sqrt(xp.mean(xp.abs(psi_all0_xp) ** 2)), backend)
    phi_max0 = _scalar(xp.max(xp.abs(psi_all0_xp)), backend)

    # Time series buffers (CPU)
    steps_trace: List[int] = [0]
    energy_trace: List[float] = [E_initial]
    norm_trace: List[float] = [N_initial]
    phi_rms_trace: List[float] = [phi_rms0]
    phi_max_trace: List[float] = [phi_max0]
    ferm_lump_trace: List[int] = [int(ferm_lumps0)]
    ferm_com_trace: List[List[float]] = [ferm_com0.tolist()]

    notes: List[str] = []

    sample_every = max(1, params.steps // 200)

    # Broadcast mass and coupling for dynamics (xp arrays)
    nb = params.n_boson
    nc = params.n_color
    ns = params.n_spin
    n_fs = nc * ns

    bos_m2_np = _broadcast_params_np(params.boson_mass2, nb, "boson_mass2")
    bos_l4_np = _broadcast_params_np(params.boson_lambda4, nb, "boson_lambda4")
    bos_m2 = xp.array(bos_m2_np).reshape((1, 1, 1, nb))
    bos_l4 = xp.array(bos_l4_np).reshape((1, 1, 1, nb))

    ferm_m2_np = _broadcast_params_np(params.fermion_mass2, n_fs, "fermion_mass2")
    ferm_l4_np = _broadcast_params_np(params.fermion_lambda4, n_fs, "fermion_lambda4")
    ferm_m2 = xp.array(ferm_m2_np).reshape((1, 1, 1, nc, ns))
    ferm_l4 = xp.array(ferm_l4_np).reshape((1, 1, 1, nc, ns))

    g_bf = float(params.g_bf)

    # Force computation (closure over xp, masses, etc.)
    def compute_forces(psi_B_local, psi_F_local):
        # Boson sector
        lapB = _laplacian_periodic(psi_B_local, xp)
        modB2 = xp.abs(psi_B_local) ** 2
        gradV_B = bos_m2 * psi_B_local + bos_l4 * modB2 * psi_B_local

        # Fermion-like sector
        lapF = _laplacian_periodic(psi_F_local, xp)
        modF2 = xp.abs(psi_F_local) ** 2
        gradV_F = ferm_m2 * psi_F_local + ferm_l4 * modF2 * psi_F_local

        # Cross-coupling
        rho_B_loc = modB2.sum(axis=-1, keepdims=True)         # (g,g,g,1)
        rho_F_loc = modF2.sum(axis=(-2, -1), keepdims=True)   # (g,g,g,1,1)

        gradV_B_cross = g_bf * psi_B_local * rho_F_loc[..., 0, 0]
        gradV_F_cross = g_bf * psi_F_local * rho_B_loc[..., 0, None]

        force_B_local = lapB - (gradV_B + gradV_B_cross)
        force_F_local = lapF - (gradV_F + gradV_F_cross)
        return force_B_local, force_F_local

    # ============================================================
    # Time evolution: leapfrog (symplectic-ish)
    # ============================================================

    # Initial half-step for momenta
    force_B0, force_F0 = compute_forces(psi_B, psi_F)
    pi_B = pi_B + 0.5 * params.dt * force_B0
    pi_F = pi_F + 0.5 * params.dt * force_F0

    for step in range(params.steps):
        # 1) Full step for psi
        psi_B = psi_B + params.dt * pi_B
        psi_F = psi_F + params.dt * pi_F

        # 2) Forces at new psi
        force_B, force_F = compute_forces(psi_B, psi_F)

        # 3) Full step for pi
        pi_B = pi_B + params.dt * force_B
        pi_F = pi_F + params.dt * force_F

        # 4) Diagnostics / sampling
        if (step + 1) % sample_every == 0 or step == params.steps - 1:
            e_step = _energy_density(psi_B, pi_B, psi_F, pi_F, params, xp, backend)
            e_mean = _scalar(e_step.mean(), backend)
            E_min = min(E_min, e_mean)
            E_max = max(E_max, e_mean)

            N_step = _compute_norm(psi_B, pi_B, psi_F, pi_F, xp, backend)

            psi_all_step = xp.concatenate(
                [
                    psi_B.reshape((*psi_B.shape[:3], -1)),
                    psi_F.reshape((*psi_F.shape[:3], -1)),
                ],
                axis=-1,
            )
            phi_rms = _scalar(
                xp.sqrt(xp.mean(xp.abs(psi_all_step) ** 2)), backend
            )
            phi_max = _scalar(xp.max(xp.abs(psi_all_step)), backend)

            rho_F_step_xp = xp.abs(psi_F) ** 2
            rho_F_step_xp = rho_F_step_xp.sum(axis=(-2, -1))
            rho_F_step = _to_numpy(rho_F_step_xp, backend)
            ferm_mask, ferm_lumps = _adaptive_lump_mask(
                rho_F_step,
                params.lump_sigma_threshold,
                params.lump_min_voxels,
            )
            ferm_com = _com_periodic(rho_F_step, ferm_mask)

            steps_trace.append(step + 1)
            energy_trace.append(e_mean)
            norm_trace.append(N_step)
            phi_rms_trace.append(phi_rms)
            phi_max_trace.append(phi_max)
            ferm_lump_trace.append(int(ferm_lumps))
            ferm_com_trace.append(ferm_com.tolist())

    # Final metrics
    e_final = _energy_density(psi_B, pi_B, psi_F, pi_F, params, xp, backend)
    E_final = _scalar(e_final.mean(), backend)
    N_final = _compute_norm(psi_B, pi_B, psi_F, pi_F, xp, backend)

    abs_energy_drift = abs(E_final - E_initial)
    rel_energy_drift = abs_energy_drift / max(1e-12, abs(E_initial))
    abs_norm_drift = abs(N_final - N_initial)
    rel_norm_drift = abs_norm_drift / max(1e-12, abs(N_initial))

    # Reasonable tolerances for this toy model
    abs_energy_tol = 5e-2
    rel_energy_tol = 1.0
    abs_norm_tol = 5e-2
    rel_norm_tol = 1.0

    energy_ok = (abs_energy_drift < abs_energy_tol) or (rel_energy_drift < rel_energy_tol)
    norm_ok = (abs_norm_drift < abs_norm_tol) or (rel_norm_drift < rel_norm_tol)
    stable = energy_ok and norm_ok

    notes: List[str] = []
    if not energy_ok:
        notes.append(f"Energy drift abs={abs_energy_drift:.3e}, rel={rel_energy_drift:.3f}")
    if not norm_ok:
        notes.append(f"Norm drift abs={abs_norm_drift:.3e}, rel={rel_norm_drift:.3f}")
    notes.append(f"backend={backend}")

    # Final fermion diagnostics
    rho_F_final_xp = xp.abs(psi_F) ** 2
    rho_F_final_xp = rho_F_final_xp.sum(axis=(-2, -1))
    rho_F_final = _to_numpy(rho_F_final_xp, backend)
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
            "phi_rms_final": phi_rms_trace[-1],
            "phi_max_final": phi_max_trace[-1],
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
