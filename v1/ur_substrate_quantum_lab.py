#!/usr/bin/env python3
"""
ur_substrate_quantum_lab.py

"Ur-script" lab tying together three layers:

1) 3D substrate test (classical field toy):
   - matter/substrate field psi with "defrag" nonlinearity g_defrag,
   - toy gauge/EM sector (A, E) with Gauss-law penalty,
   - overlap vs separation for 3D lumps,
   - optional gradient-flow evolution of the overlapping config,
   - optional substrate slice plots and CSV export.

2) Quantum CHSH test (true Hilbert-space, Bell violations):
   - two-qubit dimer with a simple entangling Hamiltonian,
   - standard CHSH measurement settings,
   - time evolution and S(t) recorded,
   - optional CSV export.

3) Fermion toy (exchange antisymmetry in Hilbert space):
   - two "positions" (sites) and two indistinguishable excitations,
   - symmetric vs antisymmetric two-particle states,
   - explicit swap operator showing +1 vs -1 eigenvalues.

This is an ongoing iterative playground, not a finished model. It gives
you one place to poke:

  1) g_defrag as an analogue of "gravity / information defrag",
  2) the Gauss/gauge sector as constraint / information routing,
  3) genuine Bell violations in a small quantum core,
  4) fermion-like antisymmetry in a minimal Hilbert-space toy.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Quantum bits (for CHSH + fermion toy)
from qutip import (
    basis,
    tensor,
    qeye,
    sigmax,
    sigmay,
    sigmaz,
    mesolve,
    expect,
    Qobj,
)

# =============================================================================
# SECTION 1: 3D SUBSTRATE (DEFRAg + GAUGE/GAUSS + OVERLAP TEST)
# =============================================================================


@dataclass
class SubstrateParams:
    # Lattice geometry
    L: int = 16
    dx: float = 1.0  # lattice spacing

    # Matter/substrate sector
    m2: float = 0.1          # mass^2 term for |psi|^2
    lambda_4: float = 1.0    # quartic self-interaction |psi|^4
    g_defrag: float = 1.0    # defragmentation nonlinearity strength

    # Gauge/EM sector
    q: float = 1.0           # coupling "charge" for covariant derivative
    c_em: float = 1.0        # EM wave speed scale (affects EM energy weights)
    g_gauge: float = 1.0     # prefactor for EM/gauge energy

    # Gauss-law constraint
    lambda_G: float = 5.0    # Gauss-law penalty strength

    # Lump configuration
    amp: float = 1.0
    sigma: float = 2.0

    # Evolution parameters (gradient flow for psi)
    dt: float = 0.01
    n_steps: int = 0  # 0 = no evolution

    # Output / visualization
    csv_out: str = ""
    plot_substrate: bool = True


@dataclass
class FieldConfig:
    """
    Container for all fields on the 3D grid:

    psi: complex matter/substrate field, shape (L, L, L)
    A:   real gauge potential, shape (3, L, L, L), components (Ax, Ay, Az)
    E:   real electric field, shape (3, L, L, L)
    """
    psi: np.ndarray
    A: np.ndarray
    E: np.ndarray


# --- finite-difference utilities ------------------------------------------------


def gradient3d(f: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the gradient of a scalar field f on a 3D periodic grid.

    Parameters
    ----------
    f : ndarray, shape (L, L, L)
        Scalar field (can be complex).
    dx : float
        Lattice spacing.

    Returns
    -------
    grad_f : ndarray, shape (3, L, L, L)
        Gradient components (df/dx, df/dy, df/dz).
    """
    dfdx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)
    dfdy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)
    dfdz = (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * dx)
    return np.stack([dfdx, dfdy, dfdz], axis=0)


def divergence3d(F: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute divergence of a vector field F on a 3D periodic grid.

    Parameters
    ----------
    F : ndarray, shape (3, L, L, L)
        Vector field components (Fx, Fy, Fz).
    dx : float
        Lattice spacing.

    Returns
    -------
    div_F : ndarray, shape (L, L, L)
        Divergence ∂x Fx + ∂y Fy + ∂z Fz.
    """
    Fx, Fy, Fz = F
    dFx_dx = (np.roll(Fx, -1, axis=0) - np.roll(Fx, 1, axis=0)) / (2.0 * dx)
    dFy_dy = (np.roll(Fy, -1, axis=1) - np.roll(Fy, 1, axis=1)) / (2.0 * dx)
    dFz_dz = (np.roll(Fz, -1, axis=2) - np.roll(Fz, 1, axis=2)) / (2.0 * dx)
    return dFx_dx + dFy_dy + dFz_dz


def curl3d(A: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the curl of a vector field A on a 3D periodic grid.

    Parameters
    ----------
    A : ndarray, shape (3, L, L, L)
        Vector potential components (Ax, Ay, Az).
    dx : float
        Lattice spacing.

    Returns
    -------
    curl_A : ndarray, shape (3, L, L, L)
        Curl components (∇×A)_i.
    """
    Ax, Ay, Az = A

    dAz_dy = (np.roll(Az, -1, axis=1) - np.roll(Az, 1, axis=1)) / (2.0 * dx)
    dAy_dz = (np.roll(Ay, -1, axis=2) - np.roll(Ay, 1, axis=2)) / (2.0 * dx)
    dAx_dz = (np.roll(Ax, -1, axis=2) - np.roll(Ax, 1, axis=2)) / (2.0 * dx)
    dAz_dx = (np.roll(Az, -1, axis=0) - np.roll(Az, 1, axis=0)) / (2.0 * dx)
    dAy_dx = (np.roll(Ay, -1, axis=0) - np.roll(Ay, 1, axis=0)) / (2.0 * dx)
    dAx_dy = (np.roll(Ax, -1, axis=1) - np.roll(Ax, 1, axis=1)) / (2.0 * dx)

    curl_x = dAz_dy - dAy_dz
    curl_y = dAx_dz - dAz_dx
    curl_z = dAy_dx - dAx_dy

    return np.stack([curl_x, curl_y, curl_z], axis=0)


# --- geometry and lumps ---------------------------------------------------------


def make_coordinate_grid(L: int, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D coordinate arrays X, Y, Z for a cubic lattice.

    Coordinates are centered so that the origin is at the lattice center.
    """
    coords_1d = (np.arange(L) - 0.5 * (L - 1)) * dx
    X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    return X, Y, Z


def gaussian_lump(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: Tuple[float, float, float],
    sigma: float,
    amp: float,
) -> np.ndarray:
    """
    Build a 3D Gaussian lump centered at 'center'.
    """
    cx, cy, cz = center
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return amp * np.exp(-0.5 * r2 / (sigma ** 2))


def build_single_lump_config(params: SubstrateParams) -> FieldConfig:
    X, Y, Z = make_coordinate_grid(params.L, params.dx)
    center = (0.0, 0.0, 0.0)
    psi_real = gaussian_lump(X, Y, Z, center, params.sigma, params.amp)
    psi = psi_real.astype(np.complex128)
    A = np.zeros((3, params.L, params.L, params.L), dtype=np.float64)
    E = np.zeros_like(A)
    return FieldConfig(psi=psi, A=A, E=E)


def build_two_lumps_overlap_config(params: SubstrateParams) -> FieldConfig:
    X, Y, Z = make_coordinate_grid(params.L, params.dx)
    center = (0.0, 0.0, 0.0)
    psi1 = gaussian_lump(X, Y, Z, center, params.sigma, params.amp)
    psi2 = gaussian_lump(X, Y, Z, center, params.sigma, params.amp)
    psi = (psi1 + psi2).astype(np.complex128)
    A = np.zeros((3, params.L, params.L, params.L), dtype=np.float64)
    E = np.zeros_like(A)
    return FieldConfig(psi=psi, A=A, E=E)


def build_two_lumps_separated_config(params: SubstrateParams) -> FieldConfig:
    X, Y, Z = make_coordinate_grid(params.L, params.dx)
    shift = 0.25 * params.L * params.dx
    center1 = (-shift, 0.0, 0.0)
    center2 = (+shift, 0.0, 0.0)

    psi1 = gaussian_lump(X, Y, Z, center1, params.sigma, params.amp)
    psi2 = gaussian_lump(X, Y, Z, center2, params.sigma, params.amp)
    psi = (psi1 + psi2).astype(np.complex128)
    A = np.zeros((3, params.L, params.L, params.L), dtype=np.float64)
    E = np.zeros_like(A)
    return FieldConfig(psi=psi, A=A, E=E)


# --- energy functionals --------------------------------------------------------


def energy_matter(config: FieldConfig, params: SubstrateParams) -> float:
    """
    Matter/substrate energy:

      E_matter = ∑ [ |D psi|^2 + m^2 |psi|^2 + λ_4 |psi|^4 ],
      D psi = (∇ - i q A) psi.
    """
    psi = config.psi
    A = config.A
    dx = params.dx

    grad_psi = gradient3d(psi, dx)
    Dpsi = grad_psi - 1j * params.q * A * psi

    kinetic = np.sum(np.abs(Dpsi) ** 2)

    density = np.abs(psi) ** 2
    mass_term = params.m2 * np.sum(density)
    quartic_term = params.lambda_4 * np.sum(density ** 2)

    return (kinetic + mass_term + quartic_term) * (dx ** 3)


def energy_defrag(config: FieldConfig, params: SubstrateParams) -> float:
    """
    Defragmentation energy:

      E_defrag = g_defrag ∑ |psi|^2 (1 - |psi|^2)^2

    This is a toy local nonlinearity that disfavors both very small and
    very large densities, encouraging moderately "clumped" structure.
    """
    psi = config.psi
    dx = params.dx
    density = np.abs(psi) ** 2
    term = density * (1.0 - density) ** 2
    return params.g_defrag * np.sum(term) * (dx ** 3)


def energy_em(config: FieldConfig, params: SubstrateParams) -> float:
    """
    Toy EM energy:

      E_EM = g_gauge ∑ [ (1/2)(|E|^2 + c_em^2 |curl A|^2) ]

    For now, we start with A=E=0, so this is zero, but the channel is here.
    """
    A = config.A
    E = config.E
    dx = params.dx

    E2 = np.sum(E ** 2)
    curlA = curl3d(A, dx)
    curlA2 = np.sum(curlA ** 2)

    return 0.5 * params.g_gauge * (E2 + (params.c_em ** 2) * curlA2) * (dx ** 3)


def energy_gauss_penalty(config: FieldConfig, params: SubstrateParams) -> float:
    """
    Gauss-law penalty:

      G = div(E) - ρ
      E_Gauss = (lambda_G / 2) ∑ G^2

    ρ is taken as |psi|^2 - mean(|psi|^2), so Gauss wants div(E) to
    match local fluctuations in density.
    """
    psi = config.psi
    E = config.E
    dx = params.dx

    density = np.abs(psi) ** 2
    rho_background = np.mean(density)
    rho = density - rho_background

    divE = divergence3d(E, dx)
    G = divE - rho

    return 0.5 * params.lambda_G * np.sum(G ** 2) * (dx ** 3)


def total_energy(config: FieldConfig, params: SubstrateParams) -> Dict[str, float]:
    E_matter = energy_matter(config, params)
    E_defrag = energy_defrag(config, params)
    E_em = energy_em(config, params)
    E_gauss = energy_gauss_penalty(config, params)
    E_total = E_matter + E_defrag + E_em + E_gauss
    return {
        "matter": E_matter,
        "defrag": E_defrag,
        "em": E_em,
        "gauss": E_gauss,
        "total": E_total,
    }


# --- gradient-flow evolution for psi ------------------------------------------


def psi_gradient_flow_step(psi: np.ndarray, params: SubstrateParams) -> np.ndarray:
    """
    One time step of simple gradient-flow evolution for psi:

      dψ/dt = ∇²ψ - m² ψ - 2 λ_4 |ψ|² ψ - g_defrag (1 - 4ρ + 3ρ²) ψ

    with ρ = |ψ|².

    This is a dissipative dynamics that tends to lower matter+defrag
    energy. Gauss/gauge are not included in this gradient yet.
    """
    dx = params.dx
    dt = params.dt

    grad_psi = gradient3d(psi, dx)
    lap_psi = divergence3d(grad_psi, dx)

    density = np.abs(psi) ** 2
    rho = density

    local_term = (
        -params.m2 * psi
        - 2.0 * params.lambda_4 * rho * psi
        - params.g_defrag * (1.0 - 4.0 * rho + 3.0 * rho ** 2) * psi
    )

    dpsi_dt = lap_psi + local_term
    psi_new = psi + dt * dpsi_dt
    return psi_new


def evolve_overlap_config(cfg_overlap: FieldConfig, params: SubstrateParams) -> FieldConfig:
    """
    Evolve the overlapping configuration under gradient flow for n_steps.
    """
    psi = cfg_overlap.psi.copy()
    for _ in range(params.n_steps):
        psi = psi_gradient_flow_step(psi, params)
    return FieldConfig(psi=psi, A=cfg_overlap.A, E=cfg_overlap.E)


# --- substrate experiment orchestration ---------------------------------------


def run_substrate_overlap_test(params: SubstrateParams) -> Dict[str, Dict[str, float]]:
    """
    Build configs:
      - one lump
      - two overlapping lumps
      - two separated lumps

    Compute energy contributions, optionally evolve the overlapping
    config, and return all diagnostics plus configs (for plotting).
    """
    cfg_one = build_single_lump_config(params)
    cfg_overlap = build_two_lumps_overlap_config(params)
    cfg_sep = build_two_lumps_separated_config(params)

    E_one = total_energy(cfg_one, params)
    E_overlap = total_energy(cfg_overlap, params)
    E_sep = total_energy(cfg_sep, params)

    results = {
        "one": E_one,
        "overlap": E_overlap,
        "sep": E_sep,
    }

    dE = E_overlap["total"] - E_sep["total"]
    dGauss = E_overlap["gauss"] - E_sep["gauss"]
    summary = {
        "ΔE_overlap_sep": dE,
        "ΔGauss_overlap_sep": dGauss,
        "E_one_total": E_one["total"],
        "E_overlap_total": E_overlap["total"],
        "E_sep_total": E_sep["total"],
    }

    cfgs = {
        "one": cfg_one,
        "overlap": cfg_overlap,
        "sep": cfg_sep,
    }

    if params.n_steps > 0:
        cfg_overlap_evolved = evolve_overlap_config(cfg_overlap, params)
        E_overlap_evolved = total_energy(cfg_overlap_evolved, params)
        results["overlap_evolved"] = E_overlap_evolved
        summary.update(
            {
                "E_overlap_evolved_total": E_overlap_evolved["total"],
                "ΔE_overlap_evolved_sep": E_overlap_evolved["total"]
                - E_sep["total"],
            }
        )
        cfgs["overlap_evolved"] = cfg_overlap_evolved

    results["summary"] = summary
    results["_configs"] = cfgs
    return results


def substrate_export_to_csv(params: SubstrateParams,
                            results: Dict[str, Dict[str, float]]) -> None:
    if not params.csv_out:
        return

    summary = results["summary"]
    one = results["one"]
    ov = results["overlap"]
    sep = results["sep"]
    ov_ev = results.get("overlap_evolved", None)

    header = [
        "L",
        "dx",
        "m2",
        "lambda_4",
        "g_defrag",
        "q",
        "c_em",
        "g_gauge",
        "lambda_G",
        "amp",
        "sigma",
        "dt",
        "n_steps",
        # one lump
        "E_one_matter",
        "E_one_defrag",
        "E_one_em",
        "E_one_gauss",
        "E_one_total",
        # overlap
        "E_ov_matter",
        "E_ov_defrag",
        "E_ov_em",
        "E_ov_gauss",
        "E_ov_total",
        # separated
        "E_sep_matter",
        "E_sep_defrag",
        "E_sep_em",
        "E_sep_gauss",
        "E_sep_total",
        # deltas
        "ΔE_overlap_sep",
        "ΔGauss_overlap_sep",
    ]

    row = [
        params.L,
        params.dx,
        params.m2,
        params.lambda_4,
        params.g_defrag,
        params.q,
        params.c_em,
        params.g_gauge,
        params.lambda_G,
        params.amp,
        params.sigma,
        params.dt,
        params.n_steps,
        one["matter"],
        one["defrag"],
        one["em"],
        one["gauss"],
        one["total"],
        ov["matter"],
        ov["defrag"],
        ov["em"],
        ov["gauss"],
        ov["total"],
        sep["matter"],
        sep["defrag"],
        sep["em"],
        sep["gauss"],
        sep["total"],
        summary["ΔE_overlap_sep"],
        summary["ΔGauss_overlap_sep"],
    ]

    if ov_ev is not None:
        header.extend(
            [
                "E_ov_ev_matter",
                "E_ov_ev_defrag",
                "E_ov_ev_em",
                "E_ov_ev_gauss",
                "E_ov_ev_total",
                "ΔE_overlap_evolved_sep",
            ]
        )
        row.extend(
            [
                ov_ev["matter"],
                ov_ev["defrag"],
                ov_ev["em"],
                ov_ev["gauss"],
                ov_ev["total"],
                summary["ΔE_overlap_evolved_sep"],
            ]
        )

    file_exists = os.path.exists(params.csv_out)
    with open(params.csv_out, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def plot_substrate_slices(results: Dict[str, Dict[str, float]],
                          params: SubstrateParams) -> None:
    """
    Plot |psi| on the central z-slice for:
      - one lump
      - overlapping
      - separated
      - (optionally) overlapping evolved
    """
    cfgs = results["_configs"]
    L = params.L
    z_idx = L // 2

    def abs_slice(cfg: FieldConfig) -> np.ndarray:
        psi = cfg.psi
        return np.abs(psi[:, :, z_idx])

    one_slice = abs_slice(cfgs["one"])
    ov_slice = abs_slice(cfgs["overlap"])
    sep_slice = abs_slice(cfgs["sep"])
    has_evolved = "overlap_evolved" in cfgs
    ov_ev_slice = abs_slice(cfgs["overlap_evolved"]) if has_evolved else None

    slices = [one_slice, ov_slice, sep_slice]
    if has_evolved:
        slices.append(ov_ev_slice)
    vmax = max(s.max() for s in slices)
    vmin = 0.0

    ncols = 4 if has_evolved else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    titles = ["One lump", "Two overlapping", "Two separated"]
    data = [one_slice, ov_slice, sep_slice]
    if has_evolved:
        titles.append("Overlap evolved")
        data.append(ov_ev_slice)

    if ncols == 1:
        axes = [axes]

    for ax, sl, title in zip(axes, data, titles):
        im = ax.imshow(
            sl.T,
            origin="lower",
            extent=[-0.5 * (L - 1), 0.5 * (L - 1), -0.5 * (L - 1), 0.5 * (L - 1)],
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("|psi| on central z-slice (3D substrate view)", fontsize=14)
    plt.tight_layout(rect=[0, 0.0, 1, 0.93])
    plt.show()


def print_substrate_summary(results: Dict[str, Dict[str, float]],
                            params: SubstrateParams) -> None:
    one = results["one"]
    ov = results["overlap"]
    sep = results["sep"]
    summary = results["summary"]
    ov_ev = results.get("overlap_evolved", None)

    print("=" * 80)
    print("3D SUBSTRATE TEST: DEFRAg + GAUGE/GAUSS + OVERLAP PENALTY")
    print("=" * 80)
    print("Substrate parameters:")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("-" * 80)

    def fmt(d: Dict[str, float]) -> str:
        return (
            f"matter={d['matter']:10.4f}, defrag={d['defrag']:10.4f}, "
            f"em={d['em']:10.4f}, gauss={d['gauss']:10.4f}, total={d['total']:10.4f}"
        )

    print("[ONE LUMP]")
    print(" ", fmt(one))
    print()
    print("[TWO LUMPS OVERLAPPING]")
    print(" ", fmt(ov))
    print()
    print("[TWO LUMPS SEPARATED]")
    print(" ", fmt(sep))
    print("-" * 80)

    dE = summary["ΔE_overlap_sep"]
    dGauss = summary["ΔGauss_overlap_sep"]
    print("[OVERLAP VS SEPARATION] (static)")
    print(f"  ΔE_total (overlap - sep)        = {dE:10.4f}")
    print(f"  ΔGauss (overlap - sep)          = {dGauss:10.4f}")
    if abs(dGauss) > 1e-12:
        print(f"  ΔE_total / ΔGauss               = {dE / dGauss:10.4f}")
    print()

    if ov_ev is not None:
        print("[OVERLAP EVOLVED UNDER GRADIENT FLOW]")
        print(" ", fmt(ov_ev))
        print(
            f"  E_overlap_evolved_total - E_sep_total = "
            f"{summary['ΔE_overlap_evolved_sep']:10.4f}"
        )
        print()

    print("Interpretation (within this toy model only):")
    print("  - Positive ΔE_total means overlap is energetically disfavored.")
    print("  - If ΔE_total tracks ΔGauss, the Gauss sector dominates overlap cost.")
    print("  - Defrag term contributes a local preference for cohesive lumps.")
    print()
    print("If the data do not show more than this, we do not say more than this.")
    print("=" * 80)


# =============================================================================
# SECTION 2: QUANTUM CHSH DIMER (BELL VIOLATIONS)
# =============================================================================


@dataclass
class CHSHParams:
    J: float = 1.0          # exchange coupling
    hz: float = 0.0         # local z-field
    t_max: float = 5.0      # max evolution time
    n_steps: int = 50       # number of time steps
    initial_state: str = "bell"  # "bell" or "product"
    csv_out: str = ""       # optional CSV path


def make_two_qubit_ops() -> Dict[str, Qobj]:
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    I = qeye(2)

    sx1 = tensor(sx, I)
    sy1 = tensor(sy, I)
    sz1 = tensor(sz, I)

    sx2 = tensor(I, sx)
    sy2 = tensor(I, sy)
    sz2 = tensor(I, sz)

    return {
        "sx1": sx1,
        "sy1": sy1,
        "sz1": sz1,
        "sx2": sx2,
        "sy2": sy2,
        "sz2": sz2,
        "I": tensor(I, I),
    }


def make_dimer_hamiltonian(params: CHSHParams, ops: Dict[str, Qobj]) -> Qobj:
    J = params.J
    hz = params.hz
    sx1, sy1, sz1 = ops["sx1"], ops["sy1"], ops["sz1"]
    sx2, sy2, sz2 = ops["sx2"], ops["sy2"], ops["sz2"]

    H_exchange = (sx1 * sx2 + sy1 * sy2 + sz1 * sz2) / 4.0
    H_field = 0.5 * hz * (sz1 + sz2)
    return J * H_exchange + H_field


def make_initial_state(params: CHSHParams) -> Qobj:
    if params.initial_state.lower() == "bell":
        up = basis(2, 0)
        down = basis(2, 1)
        phi_plus = (tensor(up, up) + tensor(down, down)).unit()
        return phi_plus
    elif params.initial_state.lower() == "product":
        up = basis(2, 0)
        return tensor(up, up)
    else:
        raise ValueError(f"Unknown initial_state: {params.initial_state}")


def make_chsh_operators(ops: Dict[str, Qobj]) -> Dict[str, Qobj]:
    sx1, sz1 = ops["sx1"], ops["sz1"]
    sx2, sz2 = ops["sx2"], ops["sz2"]

    A1 = sz1
    A1p = sx1
    B2 = (sz2 + sx2) / math.sqrt(2.0)
    B2p = (sz2 - sx2) / math.sqrt(2.0)

    S_op = (A1 * B2) + (A1 * B2p) + (A1p * B2) - (A1p * B2p)
    return {
        "A1": A1,
        "A1p": A1p,
        "B2": B2,
        "B2p": B2p,
        "S": S_op,
    }


def run_chsh_evolution(params: CHSHParams) -> Dict[str, List[float]]:
    ops = make_two_qubit_ops()
    H = make_dimer_hamiltonian(params, ops)
    psi0 = make_initial_state(params)
    chsh_ops = make_chsh_operators(ops)

    times = np.linspace(0.0, params.t_max, params.n_steps + 1)
    result = mesolve(H, psi0, times, [], [])

    A1, A1p = chsh_ops["A1"], chsh_ops["A1p"]
    B2, B2p = chsh_ops["B2"], chsh_ops["B2p"]

    S_vals = []
    E_A_B = []
    E_A_Bp = []
    E_Ap_B = []
    E_Ap_Bp = []

    for psi_t in result.states:
        E_AB = float(expect(A1 * B2, psi_t))
        E_ABp = float(expect(A1 * B2p, psi_t))
        E_ApB = float(expect(A1p * B2, psi_t))
        E_ApBp = float(expect(A1p * B2p, psi_t))
        S = E_AB + E_ABp + E_ApB - E_ApBp

        E_A_B.append(E_AB)
        E_A_Bp.append(E_ABp)
        E_Ap_B.append(E_ApB)
        E_Ap_Bp.append(E_ApBp)
        S_vals.append(S)

    return {
        "times": list(times),
        "S": S_vals,
        "E_A_B": E_A_B,
        "E_A_Bp": E_A_Bp,
        "E_Ap_B": E_Ap_B,
        "E_Ap_Bp": E_Ap_Bp,
    }


def chsh_export_to_csv(params: CHSHParams,
                       data: Dict[str, List[float]]) -> None:
    if not params.csv_out:
        return

    file_exists = os.path.exists(params.csv_out)
    with open(params.csv_out, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [
                "t",
                "S",
                "E_A_B",
                "E_A_Bp",
                "E_Ap_B",
                "E_Ap_Bp",
                "J",
                "hz",
                "t_max",
                "n_steps",
                "initial_state",
            ]
            writer.writerow(header)

        times = data["times"]
        S_vals = data["S"]
        E_A_B = data["E_A_B"]
        E_A_Bp = data["E_A_Bp"]
        E_Ap_B = data["E_Ap_B"]
        E_Ap_Bp = data["E_Ap_Bp"]

        for i, t in enumerate(times):
            writer.writerow(
                [
                    t,
                    S_vals[i],
                    E_A_B[i],
                    E_A_Bp[i],
                    E_Ap_B[i],
                    E_Ap_Bp[i],
                    params.J,
                    params.hz,
                    params.t_max,
                    params.n_steps,
                    params.initial_state,
                ]
            )


def print_chsh_summary(params: CHSHParams,
                       data: Dict[str, List[float]]) -> None:
    times = data["times"]
    S_vals = data["S"]

    max_S = max(S_vals)
    max_idx = S_vals.index(max_S)
    t_at_max = times[max_idx]

    print("=" * 80)
    print("CHSH DIMER EVOLUTION (QUANTUM CORE)")
    print("=" * 80)
    print("CHSH parameters:")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("-" * 80)
    print("Time series (first few points):")
    print("   t        S(t)")
    print("  ----------------------")
    for i in range(min(6, len(times))):
        print(f"  {times[i]:7.4f}  {S_vals[i]:7.4f}")
    print("  ...")
    print("-" * 80)
    print(f"Max S(t): {max_S:.6f} at t = {t_at_max:.6f}")
    print()
    print("Notes:")
    print("  - Local realistic theories satisfy |S| ≤ 2.")
    print("  - Quantum mechanics allows |S| up to 2√2 ≈ 2.828 (Tsirelson bound).")
    print("  - With the Bell state and these measurement settings, you should")
    print("    see S(t) near 2.828 at some times (or at t=0 for the Bell state).")
    print("=" * 80)


# =============================================================================
# SECTION 3: FERMION TOY (EXCHANGE ANTISYMMETRY)
# =============================================================================

def run_fermion_toy() -> None:
    """
    Minimal Hilbert-space toy for "fermion-like" behavior:

    Shows explicitly:
      - symmetric state → eigenvalue +1 under exchange
      - antisymmetric state → eigenvalue -1 under exchange
    """
    from qutip import Qobj, basis, tensor
    import numpy as np

    # Basis states for one qubit
    zero = basis(2, 0)
    one = basis(2, 1)

    # Two-qubit basis |x1,x2>
    ket_00 = tensor(zero, zero)
    ket_01 = tensor(zero, one)
    ket_10 = tensor(one, zero)
    ket_11 = tensor(one, one)

    # Symmetric and antisymmetric superpositions
    psi_sym = (ket_01 + ket_10).unit()
    psi_asym = (ket_01 - ket_10).unit()

    # Swap operator P_ex in the ordered basis { |00>, |01>, |10>, |11> }
    P_ex_mat = np.array(
        [
            [1, 0, 0, 0],  # |00> -> |00>
            [0, 0, 1, 0],  # |01> -> |10>
            [0, 1, 0, 0],  # |10> -> |01>
            [0, 0, 0, 1],  # |11> -> |11>
        ],
        dtype=complex,
    )

    # IMPORTANT: match tensor dims to a two-qubit operator
    P_ex = Qobj(P_ex_mat, dims=[[2, 2], [2, 2]])

    # Apply swap
    psi_sym_after = P_ex * psi_sym
    psi_asym_after = P_ex * psi_asym

    # Overlaps <ψ|P_ex|ψ> are already scalars (complex)
    overlap_sym = psi_sym.dag() * psi_sym_after
    overlap_asym = psi_asym.dag() * psi_asym_after

    overlap_sym = complex(overlap_sym)
    overlap_asym = complex(overlap_asym)

    print("=" * 80)
    print("FERMION TOY: EXCHANGE ANTISYMMETRY")
    print("=" * 80)
    print(f"<ψ_sym|P_ex|ψ_sym>   = {overlap_sym.real:+.4f} (expected +1)")
    print(f"<ψ_asym|P_ex|ψ_asym> = {overlap_asym.real:+.4f} (expected -1)")
    print()
    print("This demonstrates:")
    print("  - symmetric state → boson-like (+1)")
    print("  - antisymmetric state → fermion-like (–1)")
    print("=" * 80)

# =============================================================================
# SECTION 4: CLI / MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ur-substrate quantum lab: 3D defrag+gauge + CHSH dimer + fermion toy."
        )
    )

    # Which experiments to run
    parser.add_argument(
        "--run-substrate",
        action="store_true",
        help="Run the 3D substrate overlap test.",
    )
    parser.add_argument(
        "--no-run-substrate",
        action="store_true",
        help="Disable 3D substrate test (overrides --run-substrate).",
    )
    parser.add_argument(
        "--run-chsh",
        action="store_true",
        help="Run the two-qubit CHSH evolution.",
    )
    parser.add_argument(
        "--no-run-chsh",
        action="store_true",
        help="Disable CHSH test (overrides --run-chsh).",
    )
    parser.add_argument(
        "--run-fermion",
        action="store_true",
        help="Run the fermion exchange toy.",
    )
    parser.add_argument(
        "--no-run-fermion",
        action="store_true",
        help="Disable fermion toy (overrides --run-fermion).",
    )

    # Substrate parameters
    parser.add_argument("--L", type=int, default=16, help="Lattice size (LxLxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--m2", type=float, default=0.1, help="Mass^2 for |psi|^2 term.")
    parser.add_argument(
        "--lambda-4", type=float, default=1.0, help="Quartic |psi|^4 coupling."
    )
    parser.add_argument(
        "--g-defrag", type=float, default=1.0, help="Defragmentation coupling."
    )
    parser.add_argument("--q", type=float, default=1.0, help="Gauge charge q.")
    parser.add_argument(
        "--c-em", type=float, default=1.0, help="EM wave speed scale (energy weight)."
    )
    parser.add_argument(
        "--g-gauge", type=float, default=1.0, help="Gauge/EM sector energy prefactor."
    )
    parser.add_argument(
        "--lambda-G",
        type=float,
        default=5.0,
        help="Gauss-law penalty strength.",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=1.0,
        help="Amplitude of each Gaussian lump.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Width (sigma) of each Gaussian lump.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step for gradient-flow evolution (psi).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=0,
        help="Number of gradient-flow steps for overlapping config (0 = no evolution).",
    )
    parser.add_argument(
        "--substrate-csv-out",
        type=str,
        default="",
        help="Path to CSV file to append substrate results to (optional).",
    )
    parser.add_argument(
        "--no-plot-substrate",
        action="store_true",
        help="Disable substrate slice plots.",
    )

    # CHSH parameters
    parser.add_argument("--chsh-J", type=float, default=1.0, help="Exchange coupling J.")
    parser.add_argument("--chsh-hz", type=float, default=0.0, help="Local z-field hz.")
    parser.add_argument(
        "--chsh-t-max", type=float, default=5.0, help="CHSH max evolution time."
    )
    parser.add_argument(
        "--chsh-n-steps",
        type=int,
        default=50,
        help="CHSH steps (N) → N+1 time points including t=0.",
    )
    parser.add_argument(
        "--chsh-initial-state",
        type=str,
        default="bell",
        choices=["bell", "product"],
        help="CHSH initial state: 'bell' or 'product'.",
    )
    parser.add_argument(
        "--chsh-csv-out",
        type=str,
        default="",
        help="CSV path to write CHSH time series (optional).",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Defaults: run everything unless explicitly disabled
    run_substrate = True
    run_chsh = True
    run_fermion = True

    if args.run_substrate:
        run_substrate = True
    if args.no_run_substrate:
        run_substrate = False

    if args.run_chsh:
        run_chsh = True
    if args.no_run_chsh:
        run_chsh = False

    if args.run_fermion:
        run_fermion = True
    if args.no_run_fermion:
        run_fermion = False

    # --- Substrate experiment ---
    if run_substrate:
        sub_params = SubstrateParams(
            L=args.L,
            dx=args.dx,
            m2=args.m2,
            lambda_4=args.lambda_4,
            g_defrag=args.g_defrag,
            q=args.q,
            c_em=args.c_em,
            g_gauge=args.g_gauge,
            lambda_G=args.lambda_G,
            amp=args.amp,
            sigma=args.sigma,
            dt=args.dt,
            n_steps=args.n_steps,
            csv_out=args.substrate_csv_out,
            plot_substrate=not args.no_plot_substrate,
        )
        sub_results = run_substrate_overlap_test(sub_params)
        print_substrate_summary(sub_results, sub_params)
        substrate_export_to_csv(sub_params, sub_results)
        if sub_params.plot_substrate:
            plot_substrate_slices(sub_results, sub_params)

    # --- CHSH experiment ---
    if run_chsh:
        chsh_params = CHSHParams(
            J=args.chsh_J,
            hz=args.chsh_hz,
            t_max=args.chsh_t_max,
            n_steps=args.chsh_n_steps,
            initial_state=args.chsh_initial_state,
            csv_out=args.chsh_csv_out,
        )
        chsh_data = run_chsh_evolution(chsh_params)
        print_chsh_summary(chsh_params, chsh_data)
        chsh_export_to_csv(chsh_params, chsh_data)

    # --- Fermion toy ---
    if run_fermion:
        run_fermion_toy()


if __name__ == "__main__":
    main()
