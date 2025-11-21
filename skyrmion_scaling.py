#!/usr/bin/env python
r"""
skyrmion_scaling.py

Finite-Hilbert product-state scaling experiment for constraint-driven "exclusion strength".

We model:
- An L x L lattice of 4-level "spinor" matter sites
- Z2 gauge links (conceptually) in a uniform |+> state on each link
- A toy Hamiltonian with:
    * gauge-covariant hopping (expectation vanishes for our chosen gauge product state),
    * nearest-neighbor Sz-Sz spin coupling,
    * mass + onsite "stiffness" for occupation,
    * Gauss-law penalty enforcing local constraint consistency.

Crucially, we:
- Treat the full state as a product over sites ⊗ links.
- Compute expectation values using only local 4x4 and 2x2 matrices.
- Avoid constructing the full many-body Hilbert space, so we can scale to larger L.

For each lattice size L, we construct three configurations:
    1. One skyrmion-like pattern centered in the lattice.
    2. Two overlapping skyrmions (same center, higher amplitude).
    3. Two separated skyrmions (left and right).

For each configuration, we compute:
    - E(L): energy expectation <H>
    - N(L): total occupation
    - V(L): total Gauss-law violation sum_s <(I - G_s)^2>

We then record:
    - ΔE(L) = E_overlap(L) - E_sep(L)
    - ΔV(L) = V_overlap(L) - V_sep(L)

If ΔE(L) and ΔV(L) grow with L, it suggests that constraint-driven suppression
of overlapping patterns is not a pure finite-size artifact.

NOTE:
This is a product-state expectation calculation, not a full many-body diagonalization.
We are probing structure of the local constraint Hamiltonian on simple variational states.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Local 4x4 and 2x2 operators (matter and gauge)
# -------------------------------------------------------------

def get_matter_ops() -> Dict[str, np.ndarray]:
    """
    Returns a dict of 4x4 matter operators:
    - n_field: number (excited states have n=1)
    - sx, sy, sz: spin-like operators
    - a, a_dag: "field" ladder operators
    - n_parity: (-1)^n_field
    """
    # basis: [vac_up, vac_down, 1_up, 1_down]
    n_field = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=complex)

    sx = 0.5 * np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    sy = 0.5 * np.array([
        [0, -1j, 0, 0],
        [1j, 0, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0]
    ], dtype=complex)

    sz = 0.5 * np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)

    # ladder between vacuum and excited
    a = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=complex)
    a_dag = a.conj().T

    # parity (-1)^n: 1 on vac, -1 on excited
    n_parity = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)

    return {
        "n_field": n_field,
        "sx": sx,
        "sy": sy,
        "sz": sz,
        "a": a,
        "a_dag": a_dag,
        "n_parity": n_parity
    }


def get_gauge_ops() -> Dict[str, np.ndarray]:
    """
    Returns 2x2 gauge operators.
    We conceptually treat each link as a Z2 "gauge qubit".
    """
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=complex)
    return {"sx": sigma_x, "sz": sigma_z}


# -------------------------------------------------------------
# Skyrmion-like pattern generator on LxL lattice
# -------------------------------------------------------------

@dataclass
class SkyrmionParams:
    amplitude: float = 1.0  # base amplitude for a single skyrmion
    sigma: float = 1.0      # radial width of the Gaussian envelope


def generate_skyrmion_state(
    L: int,
    centers: List[Tuple[float, float]],
    params: SkyrmionParams
) -> np.ndarray:
    """
    Build a product-state array psi_sites[i, j, :] of shape (L, L, 4),
    representing per-site 4-component spinors.

    centers: list of (x0, y0) in lattice coordinates (0..L-1), can be fractional.
    Each center contributes a skyrmion-like excited component with:
        amplitude_k ∝ exp(-r_k^2 / (2 sigma^2)) * exp(i * theta_k)
        spin orientation encoded via a simple spinor (cos(theta_k/2), e^{i theta_k} sin(theta_k/2)).

    The local state vector at each site is:
        |psi_site> ∝ |vac_up> + sum_k amplitude_k * [spinor_up; spinor_down]
    and then normalized.
    """
    psi_sites = np.zeros((L, L, 4), dtype=complex)
    A = params.amplitude
    sigma = params.sigma

    for i in range(L):
        for j in range(L):
            # vacuum part: start as (1, 0, 0, 0)
            vac_up = 1.0 + 0.0j
            vac_down = 0.0 + 0.0j
            c_excited_up = 0.0 + 0.0j
            c_excited_down = 0.0 + 0.0j

            for (x0, y0) in centers:
                dx = i - x0
                dy = j - y0
                r2 = dx * dx + dy * dy
                r = math.sqrt(r2)
                theta = math.atan2(dy, dx) if r > 1e-12 else 0.0

                # spatial phase + radial envelope
                phase_spatial = np.exp(1j * theta)
                amp = A * np.exp(-r2 / (2.0 * sigma * sigma))

                # simple spin texture: spin angle tied to polar angle
                spin_angle = theta
                spin_up = math.cos(spin_angle / 2.0)
                spin_down = np.exp(1j * spin_angle) * math.sin(spin_angle / 2.0)

                c_excited_up += amp * phase_spatial * spin_up
                c_excited_down += amp * phase_spatial * spin_down

            # local 4-level vector: (vac_up, vac_down, 1_up, 1_down)
            v = np.array([vac_up, vac_down, c_excited_up, c_excited_down], dtype=complex)

            # normalize
            norm = math.sqrt(np.real(np.vdot(v, v)))
            if norm < 1e-14:
                # fallback to pure vacuum
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
            else:
                v = v / norm

            psi_sites[i, j, :] = v

    return psi_sites


# -------------------------------------------------------------
# Expectation value utilities for product states
# -------------------------------------------------------------

@dataclass
class HamiltonianParams:
    J_hop: float = 1.0
    J_spin: float = 0.5
    mass: float = 0.1
    U_onsite: float = 1.0
    g_gauge: float = 0.5
    lambda_G: float = 5.0


def site_expectations(
    psi_site: np.ndarray,
    ops: Dict[str, np.ndarray]
) -> Dict[str, complex]:
    """
    Compute local expectations <O> = psi^\dagger O psi for a single site.
    psi_site: shape (4,), normalized.
    ops: matter operators dict from get_matter_ops().
    """
    v = psi_site.reshape((4, 1))
    v_dag = v.conj().T

    def expec(op: np.ndarray) -> complex:
        return (v_dag @ op @ v)[0, 0]

    return {
        "n": expec(ops["n_field"]),
        "sz": expec(ops["sz"]),
        "a": expec(ops["a"]),  # <a>
        "parity": expec(ops["n_parity"])
    }


def compute_energy_and_gauss(
    psi_sites: np.ndarray,
    hparams: HamiltonianParams,
    matter_ops: Dict[str, np.ndarray]
) -> Tuple[float, float, np.ndarray]:
    """
    Compute:
      - E: expectation <H> in the product state defined by psi_sites
      - N_total: total occupation
      - V_site: array of local Gauss-law violations <(I - G_s)^2> per site

    Assumptions:
      - Gauge links are all in the |+> eigenstate of sigma_x:
            |+> = (|0> + |1>)/sqrt(2)
        so <sigma_x> = 1, <sigma_z> = 0 on every link.
      - For the Gauss operator, G_s = (-1)^{n_s} * ∏_star sigma_x = parity_s * 1.
    """
    L = psi_sites.shape[0]
    J_hop = hparams.J_hop
    J_spin = hparams.J_spin
    mass = hparams.mass
    U_onsite = hparams.U_onsite
    g_gauge = hparams.g_gauge
    lambda_G = hparams.lambda_G

    # Precompute local expectations at each site
    n = np.zeros((L, L), dtype=float)
    sz = np.zeros((L, L), dtype=float)
    a = np.zeros((L, L), dtype=complex)
    parity = np.zeros((L, L), dtype=float)

    for i in range(L):
        for j in range(L):
            ex = site_expectations(psi_sites[i, j, :], matter_ops)
            n[i, j] = np.real(ex["n"])
            sz[i, j] = np.real(ex["sz"])
            a[i, j] = ex["a"]
            parity[i, j] = np.real(ex["parity"])

    # gauge expectations for |+>: <sigma_x> = 1, <sigma_z> = 0
    sx_link = 1.0
    sz_link = 0.0

    E = 0.0

    # 1) Hopping (gauge-covariant) -- expectation vanishes for sz_link=0,
    #    but we keep it in case of future gauge choices.
    for i in range(L):
        for j in range(L):
            # right neighbor
            if j + 1 < L:
                ai = a[i, j]
                aj = a[i, j + 1]
                hop = -J_hop * 2.0 * np.real(np.conj(ai) * aj * sz_link)
                E += hop
            # down neighbor
            if i + 1 < L:
                ai = a[i, j]
                aj = a[i + 1, j]
                hop = -J_hop * 2.0 * np.real(np.conj(ai) * aj * sz_link)
                E += hop

    # 2) Spin interactions: nearest-neighbor Sz-Sz
    for i in range(L):
        for j in range(L):
            if j + 1 < L:
                E += J_spin * sz[i, j] * sz[i, j + 1]
            if i + 1 < L:
                E += J_spin * sz[i, j] * sz[i + 1, j]

    # 3) Mass + onsite "stiffness"
    N_total = 0.0
    for i in range(L):
        for j in range(L):
            n_ij = n[i, j]
            E += mass * n_ij
            E += U_onsite * (n_ij * n_ij)
            N_total += n_ij

    # 4) Gauge kinetic term: -g * sum_links <sigma_x>
    # Number of links in a 2D square lattice with open boundary:
    # horizontal links: L * (L-1), vertical links: (L-1) * L
    # total = 2 * L * (L - 1)
    n_links = 2 * L * (L - 1)
    E += -g_gauge * sx_link * n_links

    # 5) Gauss-law penalty: lambda_G * sum_s <(I - G_s)^2>
    # For G_s with eigenvalues ±1, G_s^2 = I, so operator-wise (I - G_s)^2 = 2(I - G_s).
    # Thus <(I - G_s)^2> = 2(1 - <G_s>).
    # Here <G_s> = <parity_s> * product_star <sigma_x> = parity_s * 1.
    V_site = np.zeros((L, L), dtype=float)
    if lambda_G != 0.0:
        for i in range(L):
            for j in range(L):
                G_s = parity[i, j]  # since <prod sigma_x> = 1
                V_s = 2.0 * (1.0 - G_s)
                V_site[i, j] = V_s
                E += lambda_G * V_s

    return float(E), float(N_total), V_site


# -------------------------------------------------------------
# Configuration builders for scaling experiment
# -------------------------------------------------------------

def centers_one(L: int) -> List[Tuple[float, float]]:
    """Single skyrmion centered in the lattice."""
    c = (L - 1) / 2.0
    return [(c, c)]


def centers_overlap(L: int) -> List[Tuple[float, float]]:
    """Two skyrmions overlapping at the center (effectively doubled amplitude)."""
    c = (L - 1) / 2.0
    # we will just use the same center twice; amplitudes add coherently
    return [(c, c), (c, c)]


def centers_separated(L: int) -> List[Tuple[float, float]]:
    """Two skyrmions separated horizontally."""
    c_y = (L - 1) / 2.0
    c_left = ((L - 1) / 3.0, c_y)
    c_right = (2.0 * (L - 1) / 3.0, c_y)
    return [c_left, c_right]


# -------------------------------------------------------------
# Scaling experiment driver
# -------------------------------------------------------------

def run_scaling(
    L_values: List[int],
    hparams: HamiltonianParams,
    sky_params_one: SkyrmionParams,
    sky_params_overlap: SkyrmionParams,
    sky_params_sep: SkyrmionParams
):
    matter_ops = get_matter_ops()

    print("=" * 80)
    print("SKYRMION EXCLUSION SCALING EXPERIMENT (PRODUCT-STATE EXPECTATIONS)")
    print("=" * 80)
    print(f"Hamiltonian parameters: {hparams}")
    print(f"Skyrmion params (one)      : {sky_params_one}")
    print(f"Skyrmion params (overlap)  : {sky_params_overlap}")
    print(f"Skyrmion params (separated): {sky_params_sep}")
    print()
    print("{:<4s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "L", "E1", "E_ov", "E_sep", "ΔE=Eov-Esep", "V1", "V_ov", "V_sep"
    ))
    print("-" * 100)

    results = []

    for L in L_values:
        # 1) One skyrmion
        psi_one = generate_skyrmion_state(L, centers_one(L), sky_params_one)
        E1, N1, V_site_one = compute_energy_and_gauss(psi_one, hparams, matter_ops)
        V1 = float(np.sum(V_site_one))

        # 2) Two overlapping skyrmions
        psi_ov = generate_skyrmion_state(L, centers_overlap(L), sky_params_overlap)
        E_ov, N_ov, V_site_ov = compute_energy_and_gauss(psi_ov, hparams, matter_ops)
        V_ov = float(np.sum(V_site_ov))

        # 3) Two separated skyrmions
        psi_sep = generate_skyrmion_state(L, centers_separated(L), sky_params_sep)
        E_sep, N_sep, V_site_sep = compute_energy_and_gauss(psi_sep, hparams, matter_ops)
        V_sep = float(np.sum(V_site_sep))

        dE = E_ov - E_sep
        dV = V_ov - V_sep

        print("{:<4d} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}".format(
            L, E1, E_ov, E_sep, dE, V1, V_ov, V_sep
        ))

        results.append({
            "L": L,
            "E1": E1,
            "E_ov": E_ov,
            "E_sep": E_sep,
            "dE": dE,
            "V1": V1,
            "V_ov": V_ov,
            "V_sep": V_sep,
            "dV": dV,
            "N1": N1,
            "N_ov": N_ov,
            "N_sep": N_sep
        })

    print("-" * 100)
    print("Summary (ΔE = E_overlap - E_sep, ΔV = V_overlap - V_sep):")
    for r in results:
        print(f"L={r['L']:2d}: ΔE={r['dE']:.6f}, ΔV={r['dV']:.6f}, "
              f"E1={r['E1']:.6f}, V1={r['V1']:.6f}")

    print()
    print("NOTE: These are product-state expectations in a finite model.")
    print("      If ΔE(L) and ΔV(L) increase with L, that supports the view")
    print("      that constraint-driven suppression of overlap is not purely")
    print("      a 2x2 finite-size artifact.")
    print("      If the data do not show it, we do not say it.")

    return results


# -------------------------------------------------------------
# CSV export and plotting
# -------------------------------------------------------------

def export_csv(results: List[Dict], csv_path: str) -> None:
    """
    Write scaling results to a CSV file.
    Columns: L, E1, E_ov, E_sep, dE, V1, V_ov, V_sep, dV, N1, N_ov, N_sep
    """
    fieldnames = [
        "L", "E1", "E_ov", "E_sep", "dE",
        "V1", "V_ov", "V_sep", "dV",
        "N1", "N_ov", "N_sep"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "L": r["L"],
                "E1": r["E1"],
                "E_ov": r["E_ov"],
                "E_sep": r["E_sep"],
                "dE": r["dE"],
                "V1": r["V1"],
                "V_ov": r["V_ov"],
                "V_sep": r["V_sep"],
                "dV": r["dV"],
                "N1": r["N1"],
                "N_ov": r["N_ov"],
                "N_sep": r["N_sep"],
            })
    print(f"\n[CSV] Results written to: {csv_path}")


def plot_results(results: List[Dict], lambda_G: float) -> None:
    """
    Simple plots:
      - ΔE vs L
      - ΔV vs L
    Each in its own figure.
    """
    Ls = [r["L"] for r in results]
    dEs = [r["dE"] for r in results]
    dVs = [r["dV"] for r in results]

    # Plot ΔE vs L
    plt.figure()
    plt.plot(Ls, dEs, marker="o")
    plt.xlabel("Lattice size L")
    plt.ylabel("ΔE = E_overlap - E_separated")
    plt.title(f"Overlap penalty ΔE vs L (lambda_G = {lambda_G})")
    plt.grid(True)

    # Plot ΔV vs L
    plt.figure()
    plt.plot(Ls, dVs, marker="o")
    plt.xlabel("Lattice size L")
    plt.ylabel("ΔV = V_overlap - V_separated")
    plt.title(f"Gauss violation difference ΔV vs L (lambda_G = {lambda_G})")
    plt.grid(True)

    plt.show()


# -------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Skyrmion exclusion scaling experiment in finite Hilbert space (product states)."
    )
    parser.add_argument("--L-min", type=int, default=2,
                        help="Minimum lattice size L (default: 2).")
    parser.add_argument("--L-max", type=int, default=6,
                        help="Maximum lattice size L (default: 6).")
    parser.add_argument("--lambda-G", type=float, default=5.0,
                        help="Gauss-law penalty lambda_G (default: 5.0).")
    parser.add_argument("--J-hop", type=float, default=1.0,
                        help="Hopping coefficient J_hop (default: 1.0).")
    parser.add_argument("--J-spin", type=float, default=0.5,
                        help="Spin coupling J_spin (default: 0.5).")
    parser.add_argument("--mass", type=float, default=0.1,
                        help="Mass term coefficient (default: 0.1).")
    parser.add_argument("--U-onsite", type=float, default=1.0,
                        help="Onsite stiffness U_onsite (default: 1.0).")
    parser.add_argument("--g-gauge", type=float, default=0.5,
                        help="Gauge kinetic coefficient g_gauge (default: 0.5).")
    parser.add_argument("--amp-one", type=float, default=1.0,
                        help="Base amplitude for single skyrmion (default: 1.0).")
    parser.add_argument("--amp-overlap", type=float, default=1.0,
                        help="Base amplitude per center for overlap config (default: 1.0).")
    parser.add_argument("--amp-sep", type=float, default=1.0,
                        help="Base amplitude per center for separated config (default: 1.0).")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Radial width sigma for all skyrmions (default: 1.0).")
    parser.add_argument("--csv-out", type=str, default="",
                        help="If set, write results to this CSV file.")
    parser.add_argument("--plot", action="store_true",
                        help="If set, show simple plots of ΔE and ΔV vs L.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.L_min < 2:
        raise ValueError("L_min must be at least 2.")
    if args.L_max < args.L_min:
        raise ValueError("L_max must be >= L_min.")

    L_values = list(range(args.L_min, args.L_max + 1))

    hparams = HamiltonianParams(
        J_hop=args.J_hop,
        J_spin=args.J_spin,
        mass=args.mass,
        U_onsite=args.U_onsite,
        g_gauge=args.g_gauge,
        lambda_G=args.lambda_G
    )

    sky_one = SkyrmionParams(amplitude=args.amp_one, sigma=args.sigma)
    sky_overlap = SkyrmionParams(amplitude=args.amp_overlap, sigma=args.sigma)
    sky_sep = SkyrmionParams(amplitude=args.amp_sep, sigma=args.sigma)

    results = run_scaling(L_values, hparams, sky_one, sky_overlap, sky_sep)

    if args.csv_out:
        export_csv(results, args.csv_out)

    if args.plot:
        plot_results(results, args.lambda_G)


if __name__ == "__main__":
    main()
