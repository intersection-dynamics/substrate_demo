#!/usr/bin/env python3
"""
ur_3d_defrag_gauge_overlap.py

A 3D "ur-script" that ties together:
  - a substrate/matter field with a defragmentation nonlinearity (g_defrag),
  - a toy gauge/EM sector with vector potential A and electric field E,
  - a Gauss-law penalty enforcing div(E) ~ charge density,
  - a constraint-driven overlap penalty for two 3D "skyrmion-like" lumps,
  - optional CSV export of diagnostics,
  - optional simple time evolution (gradient flow),
  - optional visualization ("little picture of the substrate").

This is NOT a full quantum many-body solver. It is a toy field-based
configuration-space model that lets us:
  - define static configurations on a 3D lattice,
  - evaluate energy contributions from matter, EM, gauge constraint,
  - compare energies for one vs overlapping vs separated lumps,
  - see how g_defrag and lambda_G participate in the overlap penalty,
  - watch an overlapping configuration relax under a simple gradient flow.
"""

import argparse
import csv
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters / data structures
# ---------------------------------------------------------------------------


@dataclass
class SimulationParams:
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

    # Skyrmion/lump configuration
    amp: float = 1.0
    sigma: float = 2.0

    # Evolution parameters
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


# ---------------------------------------------------------------------------
# 3D finite-difference utilities (periodic BCs)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Skyrmion / lump construction in 3D
# ---------------------------------------------------------------------------


def make_coordinate_grid(L: int, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D coordinate arrays X, Y, Z for a cubic lattice.

    Coordinates are centered so that the origin is at the lattice center.

    Returns
    -------
    X, Y, Z : ndarray, each shape (L, L, L)
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

    Parameters
    ----------
    X, Y, Z : ndarray
        3D coordinate arrays.
    center : tuple of float
        Center (cx, cy, cz) in the same coordinate system.
    sigma : float
        Width of the Gaussian.
    amp : float
        Amplitude.

    Returns
    -------
    psi_lump : ndarray, shape (L, L, L)
        Real-valued Gaussian profile.
    """
    cx, cy, cz = center
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return amp * np.exp(-0.5 * r2 / (sigma ** 2))


def build_single_lump_config(params: SimulationParams) -> FieldConfig:
    """
    Build a configuration with a single lump at the lattice center.
    Gauge fields are initialized to zero (no EM wave excited yet).
    """
    X, Y, Z = make_coordinate_grid(params.L, params.dx)
    center = (0.0, 0.0, 0.0)
    psi_real = gaussian_lump(X, Y, Z, center, params.sigma, params.amp)
    psi = psi_real.astype(np.complex128)
    A = np.zeros((3, params.L, params.L, params.L), dtype=np.float64)
    E = np.zeros_like(A)
    return FieldConfig(psi=psi, A=A, E=E)


def build_two_lumps_overlap_config(params: SimulationParams) -> FieldConfig:
    """
    Two lumps overlapping: both centered at the origin.
    psi = psi1 + psi2, with same center → more concentrated amplitude.
    """
    X, Y, Z = make_coordinate_grid(params.L, params.dx)
    center = (0.0, 0.0, 0.0)
    psi1 = gaussian_lump(X, Y, Z, center, params.sigma, params.amp)
    psi2 = gaussian_lump(X, Y, Z, center, params.sigma, params.amp)
    psi = (psi1 + psi2).astype(np.complex128)
    A = np.zeros((3, params.L, params.L, params.L), dtype=np.float64)
    E = np.zeros_like(A)
    return FieldConfig(psi=psi, A=A, E=E)


def build_two_lumps_separated_config(params: SimulationParams) -> FieldConfig:
    """
    Two lumps separated: one near +x corner, one near -x corner.
    This mimics two skyrmion-like excitations far apart.
    """
    X, Y, Z = make_coordinate_grid(params.L, params.dx)

    # Place lumps at two separated positions along x-axis
    shift = 0.25 * params.L * params.dx
    center1 = (-shift, 0.0, 0.0)
    center2 = (+shift, 0.0, 0.0)

    psi1 = gaussian_lump(X, Y, Z, center1, params.sigma, params.amp)
    psi2 = gaussian_lump(X, Y, Z, center2, params.sigma, params.amp)
    psi = (psi1 + psi2).astype(np.complex128)
    A = np.zeros((3, params.L, params.L, params.L), dtype=np.float64)
    E = np.zeros_like(A)
    return FieldConfig(psi=psi, A=A, E=E)


# ---------------------------------------------------------------------------
# Energy functionals
# ---------------------------------------------------------------------------


def energy_matter(config: FieldConfig, params: SimulationParams) -> float:
    """
    Matter/substrate energy:

    E_matter = ∑ [ |D psi|^2 + m^2 |psi|^2 + λ_4 |psi|^4 ]
      where D psi = (∇ - i q A) psi

    This is a toy; we ignore time derivatives and just evaluate static energy.
    """
    psi = config.psi
    A = config.A
    dx = params.dx

    # Spatial gradient of psi (component-wise)
    grad_psi = gradient3d(psi, dx)  # shape (3, L, L, L), complex

    # Covariant derivative: D psi = (∇ - i q A) psi
    Dpsi = grad_psi - 1j * params.q * A * psi

    # |D psi|^2
    kinetic = np.sum(np.abs(Dpsi) ** 2)

    # m^2 |psi|^2 term
    density = np.abs(psi) ** 2
    mass_term = params.m2 * np.sum(density)

    # λ_4 |psi|^4 term
    quartic_term = params.lambda_4 * np.sum(density ** 2)

    return (kinetic + mass_term + quartic_term) * (dx ** 3)


def energy_defrag(config: FieldConfig, params: SimulationParams) -> float:
    """
    Defragmentation energy.

    Here we pick a simple local nonlinearity that favours "clumped"
    amplitude away from 0 and away from very fragmented extremes:

      E_defrag = g_defrag ∑ |psi|^2 (1 - |psi|^2)^2

    In regions where |psi|^2 ~ 0 or very large, this term grows,
    encouraging a moderate, cohesive blob structure.
    """
    psi = config.psi
    dx = params.dx
    density = np.abs(psi) ** 2
    term = density * (1.0 - density) ** 2
    return params.g_defrag * np.sum(term) * (dx ** 3)


def energy_em(config: FieldConfig, params: SimulationParams) -> float:
    """
    Toy EM energy:

      E_EM = g_gauge ∑ [ (1/2)(|E|^2 + c_em^2 |curl A|^2) ]

    This encodes:
      - electric field energy ~ |E|^2
      - magnetic-like energy via curl(A)

    For the static lump tests we start with A=0, E=0, so E_EM=0.
    But the term is here as part of the "ur-script" layout.
    """
    A = config.A
    E = config.E
    dx = params.dx

    E2 = np.sum(E ** 2)
    curlA = curl3d(A, dx)
    curlA2 = np.sum(curlA ** 2)

    return 0.5 * params.g_gauge * (E2 + (params.c_em ** 2) * curlA2) * (dx ** 3)


def energy_gauss_penalty(config: FieldConfig, params: SimulationParams) -> float:
    """
    Gauss-law penalty:

      G = div(E) - ρ
      E_Gauss = (lambda_G / 2) ∑ G^2

    where we take a simple "charge density" ρ = |psi|^2 - ρ_background.

    For the background, we use the spatial average of |psi|^2,
    so that Gauss wants div(E) to match the local fluctuation away
    from the mean density.

    This is a toy version of constraint-driven overlap suppression:
      - Overlapping lumps create a more concentrated ρ,
      - which drives larger Gauss penalty unless E rearranges.
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


def total_energy(config: FieldConfig, params: SimulationParams) -> Dict[str, float]:
    """
    Compute all sector energies and total.
    """
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


# ---------------------------------------------------------------------------
# Simple gradient-flow evolution for psi
# ---------------------------------------------------------------------------


def psi_gradient_flow_step(psi: np.ndarray, params: SimulationParams) -> np.ndarray:
    """
    One time step of a simple gradient-flow evolution for psi:

      dψ/dt = ∇²ψ - m² ψ - 2 λ_4 |ψ|² ψ - g_defrag (1 - 4ρ + 3ρ²) ψ

    with ρ = |ψ|².

    This is a dissipative dynamics that tends to decrease the
    matter+defrag energy. Gauge/EM terms are ignored here (A=E=0).
    """
    dx = params.dx
    dt = params.dt

    # Laplacian via divergence of gradient (periodic BCs)
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


def evolve_overlap_config(
    cfg_overlap: FieldConfig, params: SimulationParams
) -> FieldConfig:
    """
    Evolve the overlapping configuration psi under gradient flow
    for n_steps with step size dt.

    Returns
    -------
    FieldConfig
        New config with evolved psi (A and E unchanged).
    """
    psi = cfg_overlap.psi.copy()
    for _ in range(params.n_steps):
        psi = psi_gradient_flow_step(psi, params)
    return FieldConfig(psi=psi, A=cfg_overlap.A, E=cfg_overlap.E)


# ---------------------------------------------------------------------------
# Diagnostics: overlap penalty in 3D
# ---------------------------------------------------------------------------


def run_overlap_test(params: SimulationParams) -> Dict[str, Dict[str, float]]:
    """
    Build three configurations:

      1) One lump
      2) Two lumps overlapping (same center)
      3) Two lumps separated (far apart)

    Compute energy contributions for each, and return a dict:

      {
        "one": {...},
        "overlap": {...},
        "sep": {...},
        "summary": {...}
      }

    If params.n_steps > 0, we also evolve the overlapping configuration
    under gradient flow and report its energy as "overlap_evolved".
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

    # Direct overlap penalty vs separation (static)
    dE = E_overlap["total"] - E_sep["total"]
    dGauss = E_overlap["gauss"] - E_sep["gauss"]

    summary = {
        "ΔE_overlap_sep": dE,
        "ΔGauss_overlap_sep": dGauss,
        "E_one_total": E_one["total"],
        "E_overlap_total": E_overlap["total"],
        "E_sep_total": E_sep["total"],
    }

    # Optional evolution of overlapping config
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

        # Store the evolved config so we can plot it later if we want
        results["_configs"] = {
            "one": cfg_one,
            "overlap": cfg_overlap,
            "overlap_evolved": cfg_overlap_evolved,
            "sep": cfg_sep,
        }
    else:
        results["_configs"] = {
            "one": cfg_one,
            "overlap": cfg_overlap,
            "sep": cfg_sep,
        }

    results["summary"] = summary
    return results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_to_csv(
    params: SimulationParams, results: Dict[str, Dict[str, float]], path: str
):
    """
    Append one line of results to a CSV file.

    Columns include:
      - basic params,
      - energies for one/overlap/sep (and overlap_evolved if present),
      - ΔE and ΔGauss diagnostics.
    """
    if not path:
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
        # Extend header/row just in case you later care about it
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

    # If file doesn't exist, write header; else append
    file_exists = os.path.exists(path)
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Visualization ("little picture of the substrate")
# ---------------------------------------------------------------------------


def plot_substrate_slices(results: Dict[str, Dict[str, float]], params: SimulationParams):
    """
    Plot |psi| on the central z-slice for:

      - one lump
      - two overlapping lumps (initial)
      - two separated lumps
      - (optional) overlapping evolved

    This gives a "little picture" of the substrate configurations.
    """
    cfgs = results["_configs"]
    L = params.L
    z_idx = L // 2  # central slice

    def abs_slice(cfg: FieldConfig) -> np.ndarray:
        psi = cfg.psi
        return np.abs(psi[:, :, z_idx])

    one_slice = abs_slice(cfgs["one"])
    ov_slice = abs_slice(cfgs["overlap"])
    sep_slice = abs_slice(cfgs["sep"])
    ov_ev_slice = None
    has_evolved = "overlap_evolved" in cfgs
    if has_evolved:
        ov_ev_slice = abs_slice(cfgs["overlap_evolved"])

    # Determine color scale based on max across all slices
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


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args() -> SimulationParams:
    parser = argparse.ArgumentParser(
        description=(
            "3D ur-script: defrag + gauge/EM + Gauss + overlap penalty "
            "+ optional CSV export + optional evolution + visualization."
        )
    )
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
        "--csv-out",
        type=str,
        default="",
        help="Path to CSV file to append results to (optional).",
    )
    parser.add_argument(
        "--no-plot-substrate",
        action="store_true",
        help="Disable substrate slice plots.",
    )

    args = parser.parse_args()
    return SimulationParams(
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
        csv_out=args.csv_out,
        plot_substrate=not args.no_plot_substrate,
    )


def print_energy_table(results: Dict[str, Dict[str, float]], params: SimulationParams):
    one = results["one"]
    ov = results["overlap"]
    sep = results["sep"]
    summary = results["summary"]
    ov_ev = results.get("overlap_evolved", None)

    print("=" * 80)
    print("3D UR-SCRIPT: DEFRAg + GAUGE/EM + GAUSS + OVERLAP PENALTY")
    print("=" * 80)
    print("Parameters:")
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


def main():
    params = parse_args()
    results = run_overlap_test(params)
    print_energy_table(results, params)

    # CSV export
    if params.csv_out:
        export_to_csv(params, results, params.csv_out)

    # Plot substrate slices (including evolved overlap if present)
    if params.plot_substrate:
        plot_substrate_slices(results, params)


if __name__ == "__main__":
    main()
