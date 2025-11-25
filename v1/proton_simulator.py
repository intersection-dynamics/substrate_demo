#!/usr/bin/env python3
"""
proton_simulator.py

Toy 3-quark "proton" simulator built on top of the 3D substrate ideas.

- Three spin-1/2 excitations ("quarks") on a 3D Lx x Ly x Lz lattice.
- Degrees of freedom: positions r1,r2,r3 and spins s1,s2,s3 (↑,↓).
- Basis: |r1,s1; r2,s2; r3,s3>, dimension dim = (Ns*2)^3, Ns = Lx*Ly*Lz.

Hamiltonian (schematic):

    H = H_hop + H_mass + H_defrag + H_Gauss + H_contact + H_Zeeman

where
  - H_hop      : nearest-neighbor hopping for each quark (periodic BC).
  - H_mass     : mass term m per quark.
  - H_defrag   : central "clumping" potential to bind quarks.
  - H_Gauss    : Gauss-like occupancy penalty around rho0 = 3/Ns.
  - H_contact  : local spin interaction for each pair at same site,
                 similar in spirit to the two-fermion substrate:
                 Heisenberg-like Sz·Sz + singlet bonus and flip mixing.
  - H_Zeeman   : -B * (S_z1 + S_z2 + S_z3), which selects a definite S_z
                 when B ≠ 0 (e.g. pick S_z = +1/2 for a "proton" doublet).

This is NOT a realistic QCD proton; it's a substrate-based 3-body
bound state that lets you study how three excitations clump, how
their spins align, and what the effective "proton" ground state
looks like on your substrate.

Usage (from project folder):

    python proton_simulator.py

You can also tweak parameters, e.g.:

    python proton_simulator.py --lambda_G 8.0 --J_exch 1.5 --sigma_defrag 0.8 --B 0.1
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# Reuse 3D lattice helpers from the two-fermion engine
from substrate_engine_3d import site_index_3d, site_coords_3d, build_neighbors_3d


# =============================================================================
# Parameters / dataclass
# =============================================================================

@dataclass
class ThreeQuarkParams:
    # Lattice
    Lx: int = 2
    Ly: int = 2
    Lz: int = 2

    # Single-particle / substrate
    J_hop: float = 1.0
    m: float = 0.1
    g_defrag: float = 1.0
    sigma_defrag: float = 1.0

    # Gauss-like constraint and spin sector
    lambda_G: float = 5.0
    lambda_S: float = -1.0
    lambda_T: float = 0.0
    J_exch: float = 1.0

    # Zeeman field (selects S_z if nonzero)
    B_field: float = 0.0

    # Solver
    max_eigsh_iter: int = 20000
    k_eigs: int = 1


# =============================================================================
# Basis encoding / decoding
# =============================================================================

def encode_basis_3q(
    r1: int, s1: int,
    r2: int, s2: int,
    r3: int, s3: int,
    Ns: int
) -> int:
    """
    Encode 3-quark basis index:

        idx = ((((r1*2+s1) * Ns*2) + (r2*2+s2)) * Ns*2) + (r3*2+s3)

    s1,s2,s3 in {0,1} for (↑,↓).
    """
    return (((r1 * 2 + s1) * Ns * 2 + (r2 * 2 + s2)) * Ns * 2 + (r3 * 2 + s3))


def decode_basis_3q(idx: int, Ns: int) -> Tuple[int, int, int, int, int, int]:
    """
    Inverse mapping of encode_basis_3q.
    """
    tmp = idx
    r3s3 = tmp % (Ns * 2)
    tmp //= (Ns * 2)
    r2s2 = tmp % (Ns * 2)
    tmp //= (Ns * 2)
    r1s1 = tmp

    r1 = r1s1 // 2
    s1 = r1s1 % 2
    r2 = r2s2 // 2
    s2 = r2s2 % 2
    r3 = r3s3 // 2
    s3 = r3s3 % 2
    return r1, s1, r2, s2, r3, s3


# =============================================================================
# Substrate potentials / Gauss term
# =============================================================================

def defrag_potential_3q(params: ThreeQuarkParams) -> np.ndarray:
    """
    3D Gaussian defrag potential centered in the box:

        V_defrag(r) = -exp(-|r - r0|^2 / (2 sigma^2))

    defined on sites r = 0..Ns-1.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    sigma = params.sigma_defrag
    Ns = Lx * Ly * Lz

    V = np.zeros(Ns, dtype=float)
    if sigma <= 0.0:
        return V

    cx = 0.5 * (Lx - 1)
    cy = 0.5 * (Ly - 1)
    cz = 0.5 * (Lz - 1)

    for r in range(Ns):
        x, y, z = site_coords_3d(r, Lx, Ly, Lz)
        dx = x - cx
        dy = y - cy
        dz = z - cz
        r2 = dx * dx + dy * dy + dz * dz
        V[r] = -np.exp(-0.5 * r2 / (sigma * sigma))

    return V


def gauss_penalty_energy(
    occ: np.ndarray,
    params: ThreeQuarkParams
) -> float:
    """
    Simple Gauss-like occupancy penalty:

        E_Gauss = 0.5 * lambda_G * sum_r (occ[r] - rho0)^2

    with rho0 = 3 / Ns (three quarks spread over Ns sites on average).
    """
    Ns = occ.shape[0]
    rho0 = 3.0 / Ns
    G = occ.astype(float) - rho0
    return 0.5 * params.lambda_G * np.sum(G * G)


# =============================================================================
# Hamiltonian construction
# =============================================================================

def build_threequark_hamiltonian(params: ThreeQuarkParams) -> csr_matrix:
    """
    Build the 3D three-quark Hamiltonian:

        H = H_hop + H_mass + H_defrag + H_Gauss + H_contact + H_Zeeman

    as a sparse CSR matrix.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = (Ns * 2) ** 3

    neighbors = build_neighbors_3d(Lx, Ly, Lz)
    V_defrag = defrag_potential_3q(params)

    H = lil_matrix((dim, dim), dtype=np.complex128)

    for idx in range(dim):
        r1, s1, r2, s2, r3, s3 = decode_basis_3q(idx, Ns)

        # 1) mass: m per quark
        H[idx, idx] += 3.0 * params.m

        # 2) defrag potential: g_defrag * sum_i V_defrag[ri]
        H[idx, idx] += params.g_defrag * (V_defrag[r1] + V_defrag[r2] + V_defrag[r3])

        # 3) Gauss-like occupancy penalty
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        occ[r3] += 1
        H[idx, idx] += gauss_penalty_energy(occ, params)

        # 4) Local spin contact terms for each pair at the same site
        #    Heisenberg-like Sz·Sz + singlet bonus + flip mixing.

        def pair_contact(
            r_a: int, s_a: int,
            r_b: int, s_b: int,
            r_other: int, s_other: int,
            which_pair: str
        ):
            # Only if they sit on the same site
            if r_a != r_b:
                return

            # S^z = +1/2 for s=0 (↑), -1/2 for s=1 (↓)
            sz_a = +0.5 if s_a == 0 else -0.5
            sz_b = +0.5 if s_b == 0 else -0.5

            # Heisenberg diagonal piece
            H[idx, idx] += params.J_exch * (sz_a * sz_b)

            # Triplet / singlet shifts
            if s_a == s_b:
                # Triplet sector (↑↑ or ↓↓)
                H[idx, idx] += params.lambda_T
            else:
                # Singlet sector (↑↓ or ↓↑)
                H[idx, idx] += params.lambda_S

                # Spin-flip mixing |↑↓> <-> |↓↑> for this pair
                s_a_p, s_b_p = s_b, s_a

                if which_pair == "12":
                    idx_flip = encode_basis_3q(
                        r1, s_a_p,
                        r2, s_b_p,
                        r_other, s_other,
                        Ns,
                    )
                elif which_pair == "13":
                    idx_flip = encode_basis_3q(
                        r1, s_a_p,
                        r_other, s_other,
                        r3, s_b_p,
                        Ns,
                    )
                else:  # "23"
                    idx_flip = encode_basis_3q(
                        r_other, s_other,
                        r2, s_a_p,
                        r3, s_b_p,
                        Ns,
                    )

                H[idx_flip, idx] += 0.5 * params.J_exch

        # Apply to pairs (1,2), (1,3), (2,3)
        pair_contact(r1, s1, r2, s2, r3, s3, "12")
        pair_contact(r1, s1, r3, s3, r2, s2, "13")
        pair_contact(r2, s2, r3, s3, r1, s1, "23")

        # 5) Hopping for each quark
        # particle 1
        for r1p in neighbors[r1]:
            idx_new = encode_basis_3q(r1p, s1, r2, s2, r3, s3, Ns)
            H[idx_new, idx] += -params.J_hop

        # particle 2
        for r2p in neighbors[r2]:
            idx_new = encode_basis_3q(r1, s1, r2p, s2, r3, s3, Ns)
            H[idx_new, idx] += -params.J_hop

        # particle 3
        for r3p in neighbors[r3]:
            idx_new = encode_basis_3q(r1, s1, r2, s2, r3p, s3, Ns)
            H[idx_new, idx] += -params.J_hop

        # 6) Zeeman term: -B * (S_z1 + S_z2 + S_z3)
        if params.B_field != 0.0:
            sz1 = +0.5 if s1 == 0 else -0.5
            sz2 = +0.5 if s2 == 0 else -0.5
            sz3 = +0.5 if s3 == 0 else -0.5
            H[idx, idx] += -params.B_field * (sz1 + sz2 + sz3)

    return H.tocsr()


# =============================================================================
# Ground state and basic observables
# =============================================================================

def compute_ground_state(params: ThreeQuarkParams) -> Dict[str, Any]:
    """
    Build H, compute ground state via eigsh, and extract basic diagnostics.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = (Ns * 2) ** 3

    H = build_threequark_hamiltonian(params)

    print(f"Building 3-quark Hamiltonian: dim = {dim}")
    print("Diagonalizing ground state with eigsh...")

    evals, evecs = eigsh(H, k=params.k_eigs, which="SA", maxiter=params.max_eigsh_iter)
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    # Normalize explicitly
    norm = np.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    # Basic observables
    obs = analyze_proton_state(params, psi0)

    result = {
        "E0": E0,
        "dim": dim,
        **obs,
    }
    return result


def analyze_proton_state(params: ThreeQuarkParams, psi: np.ndarray) -> Dict[str, Any]:
    """
    Compute quark density, average pair separation, total spin stats,
    and a proton-like radius from the ground state wavefunction.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = psi.shape[0]

    # Site occupancy distribution
    occ = np.zeros(Ns, dtype=float)

    # For radius: accumulate (barycenter, r^2 wrt barycenter)
    # We'll first compute average position of each quark, then recompute <r^2>.
    # But on this tiny lattice, an approximate barycenter using occupancy alone
    # is already meaningful.

    # Pair distance accumulators
    d12_acc = 0.0
    d13_acc = 0.0
    d23_acc = 0.0

    # Total spin-z statistics
    sz_tot_acc = 0.0
    sz_tot_sq_acc = 0.0

    # Probabilities for S_z = 3/2, 1/2, -1/2, -3/2
    sz_hist: Dict[float, float] = {1.5: 0.0, 0.5: 0.0, -0.5: 0.0, -1.5: 0.0}

    # For barycenter: accumulate position-weighted occupancy
    pos_weighted = np.zeros(3, dtype=float)
    total_quark_weight = 0.0

    # First pass: observables and occupancy
    for idx in range(dim):
        amp = psi[idx]
        p = float(np.abs(amp) ** 2)
        if p < 1e-14:
            continue

        r1, s1, r2, s2, r3, s3 = decode_basis_3q(idx, Ns)

        # Occupancy and position weighting
        for r in (r1, r2, r3):
            occ[r] += p
            x, y, z = site_coords_3d(r, Lx, Ly, Lz)
            pos_weighted += p * np.array([x, y, z], dtype=float)
            total_quark_weight += p

        # Positions
        x1, y1, z1 = site_coords_3d(r1, Lx, Ly, Lz)
        x2, y2, z2 = site_coords_3d(r2, Lx, Ly, Lz)
        x3, y3, z3 = site_coords_3d(r3, Lx, Ly, Lz)

        # Pairwise distances (Euclidean, non-periodic for now)
        d12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        d13 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2 + (z1 - z3) ** 2)
        d23 = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2)

        d12_acc += p * d12
        d13_acc += p * d13
        d23_acc += p * d23

        # Spin z components
        sz1 = +0.5 if s1 == 0 else -0.5
        sz2 = +0.5 if s2 == 0 else -0.5
        sz3 = +0.5 if s3 == 0 else -0.5
        sz_tot = sz1 + sz2 + sz3

        sz_tot_acc += p * sz_tot
        sz_tot_sq_acc += p * (sz_tot ** 2)

        # Bin into histogram
        for key in sz_hist.keys():
            if abs(sz_tot - key) < 1e-8:
                sz_hist[key] += p
                break

    # Normalize occ so sum over sites = average number of quarks (should be 3)
    occ_sum = np.sum(occ)
    if occ_sum > 0:
        occ_norm = occ * (3.0 / occ_sum)
    else:
        occ_norm = occ

    avg_d12 = d12_acc
    avg_d13 = d13_acc
    avg_d23 = d23_acc

    sz_tot_exp = sz_tot_acc
    sz_tot_sq_exp = sz_tot_sq_acc
    sz_hist = {float(k): float(v) for k, v in sz_hist.items()}

    # Barycenter estimate
    if total_quark_weight > 0:
        barycenter = pos_weighted / (3.0 * total_quark_weight)
    else:
        barycenter = np.array([0.0, 0.0, 0.0], dtype=float)

    # Second pass: compute <r^2> relative to barycenter
    r2_acc = 0.0
    for idx in range(dim):
        amp = psi[idx]
        p = float(np.abs(amp) ** 2)
        if p < 1e-14:
            continue

        r1, s1, r2, s2, r3, s3 = decode_basis_3q(idx, Ns)
        for r in (r1, r2, r3):
            x, y, z = site_coords_3d(r, Lx, Ly, Lz)
            dx, dy, dz = x - barycenter[0], y - barycenter[1], z - barycenter[2]
            r2_acc += p * (dx * dx + dy * dy + dz * dz)

    # <r^2> per quark
    r2_per_quark = r2_acc / 3.0
    proton_radius = np.sqrt(r2_per_quark)

    return {
        "occ_per_site": occ_norm.tolist(),
        "avg_pair_distance_12": avg_d12,
        "avg_pair_distance_13": avg_d13,
        "avg_pair_distance_23": avg_d23,
        "S_z_expectation": sz_tot_exp,
        "S_z2_expectation": sz_tot_sq_exp,
        "S_z_histogram": sz_hist,
        "barycenter": barycenter.tolist(),
        "r2_per_quark": r2_per_quark,
        "effective_proton_radius": proton_radius,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Toy 3-quark proton simulator on the 3D substrate."
    )
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=2)
    parser.add_argument("--Lz", type=int, default=2)
    parser.add_argument("--J_hop", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--g_defrag", type=float, default=1.0)
    parser.add_argument("--sigma_defrag", type=float, default=1.0)
    parser.add_argument("--lambda_G", type=float, default=5.0)
    parser.add_argument("--lambda_S", type=float, default=-1.0)
    parser.add_argument("--lambda_T", type=float, default=0.0)
    parser.add_argument("--J_exch", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=0.0,
                        help="Zeeman field strength (select S_z if nonzero).")
    parser.add_argument("--max_iter", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)

    params = ThreeQuarkParams(
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        J_hop=args.J_hop,
        m=args.m,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        lambda_G=args.lambda_G,
        lambda_S=args.lambda_S,
        lambda_T=args.lambda_T,
        J_exch=args.J_exch,
        B_field=args.B,
        max_eigsh_iter=args.max_iter,
        k_eigs=1,
    )

    print("\n" + "=" * 70)
    print("3-QUARK PROTON SIMULATOR (TOY SUBSTRATE MODEL)")
    print("=" * 70)
    print("Parameters:")
    for k, v in asdict(params).items():
        print(f"  {k:12s} = {v}")
    print("=" * 70)

    result = compute_ground_state(params)

    print("\nGround state summary:")
    print(f"  E0                = {result['E0']:.6f}")
    print(f"  dim               = {result['dim']}")
    print(f"  <S_z>             = {result['S_z_expectation']:.6f}")
    print(f"  <S_z^2>           = {result['S_z2_expectation']:.6f}")
    print(f"  <d_12>            = {result['avg_pair_distance_12']:.6f}")
    print(f"  <d_13>            = {result['avg_pair_distance_13']:.6f}")
    print(f"  <d_23>            = {result['avg_pair_distance_23']:.6f}")
    print(f"  r_p (eff radius)  = {result['effective_proton_radius']:.6f}")
    print("  S_z histogram     =")
    for sz, p in sorted(result["S_z_histogram"].items(), reverse=True):
        print(f"    S_z = {sz:+.1f} : {p:.4f}")

    bx, by, bz = result["barycenter"]
    print(f"\nBarycenter (x,y,z): ({bx:.3f}, {by:.3f}, {bz:.3f})")

    print("\nSite occupancy (normalized to 3 quarks total):")
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    occ = np.array(result["occ_per_site"])
    for r in range(Ns):
        x, y, z = site_coords_3d(r, Lx, Ly, Lz)
        print(f"  r={r:2d} (x,y,z)=({x},{y},{z}) : n(r) ≈ {occ[r]:.4f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
