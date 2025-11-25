#!/usr/bin/env python3
"""
substrate_engine_3d.py

3D finite-Hilbert substrate with TWO spin-1/2 excitations on a 3D lattice.

- Lattice: Lx x Ly x Lz sites, Ns = Lx*Ly*Lz.
- Degrees of freedom: positions r1, r2 (sites), spins s1, s2 (↑/↓).
- Basis states: |r1, s1; r2, s2>, with r in {0,...,Ns-1}, s in {0,1}.
- Hilbert dimension: dim = (Ns * 2) * (Ns * 2).

Hamiltonian:
    H = H_hop + H_mass + H_defrag + H_Gauss + H_contact

where:
  - H_hop      : nearest-neighbor hopping (3D periodic BC).
  - H_mass     : effective mass term (onsite).
  - H_defrag   : nonlocal "defragmentation" / attraction based on Gaussian
                 convolution of densities.
  - H_Gauss    : Gauss-like "radial" operator favoring localized lumps.
  - H_contact  : onsite contact terms for spin interaction / exclusion.

Diagnostics include:
  - Exchange antisymmetry (r1,s1) <-> (r2,s2)
  - Overlap + spin structure (singlet/triplet at r1==r2)
  - Gauss-like energy expectation
  - Site-wise lump densities
  - CHSH Bell inequality in the 2-spin reduced state
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


# =============================================================================
# Parameters and indexing
# =============================================================================


@dataclass
class TwoFermion3DParams:
    """
    Parameters for the 3D two-excitation substrate model.
    """
    Lx: int = 2
    Ly: int = 2
    Lz: int = 2

    J_hop: float = 1.0
    mass: float = 0.1

    g_defrag: float = 1.0
    sigma_defrag: float = 1.0

    lambda_G: float = 5.0
    lambda_S: float = -1.0
    lambda_T: float = 0.0
    J_exch: float = 1.0

    max_eigsh_iter: int = 5000
    k_eigs: int = 1


def site_index_3d(x: int, y: int, z: int, Lx: int, Ly: int, Lz: int) -> int:
    """
    Map 3D coordinates (x,y,z) to a single site index in [0, Lx*Ly*Lz).
    """
    return x + Lx * (y + Ly * z)


def decode_site_3d(r: int, Lx: int, Ly: int, Lz: int) -> Tuple[int, int, int]:
    """
    Inverse of site_index_3d.
    """
    z = r // (Lx * Ly)
    rem = r % (Lx * Ly)
    y = rem // Lx
    x = rem % Lx
    return x, y, z


def encode_basis_3d(r1: int, s1: int, r2: int, s2: int, Ns: int) -> int:
    """
    Encode (r1,s1,r2,s2) into a single Hilbert-space index in [0, Ns*2*Ns*2).
    """
    return ((r1 * 2 + s1) * Ns * 2) + (r2 * 2 + s2)


def decode_basis_3d(idx: int, Ns: int) -> Tuple[int, int, int, int]:
    """
    Decode a basis index in [0, Ns*2*Ns*2) into (r1,s1,r2,s2).
    """
    r1s1, r2s2 = divmod(idx, Ns * 2)
    r1, s1 = divmod(r1s1, 2)
    r2, s2 = divmod(r2s2, 2)
    return r1, s1, r2, s2


# =============================================================================
# Neighbor list and defrag potential
# =============================================================================


def build_neighbors_3d(Lx: int, Ly: int, Lz: int) -> Dict[int, list]:
    """
    Build a dict mapping each site index r to a list of nearest neighbors,
    using 3D periodic boundary conditions.
    """
    Ns = Lx * Ly * Lz
    neighbors: Dict[int, list] = {r: [] for r in range(Ns)}

    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                r = site_index_3d(x, y, z, Lx, Ly, Lz)

                xp = (x + 1) % Lx
                xm = (x - 1) % Lx
                yp = (y + 1) % Ly
                ym = (y - 1) % Ly
                zp = (z + 1) % Lz
                zm = (z - 1) % Lz

                neighbors[r].append(site_index_3d(xp, y, z, Lx, Ly, Lz))
                neighbors[r].append(site_index_3d(xm, y, z, Lx, Ly, Lz))
                neighbors[r].append(site_index_3d(x, yp, z, Lx, Ly, Lz))
                neighbors[r].append(site_index_3d(x, ym, z, Lx, Ly, Lz))
                neighbors[r].append(site_index_3d(x, y, zp, Lx, Ly, Lz))
                neighbors[r].append(site_index_3d(x, y, zm, Lx, Ly, Lz))

    return neighbors


def defrag_potential_3d(params: TwoFermion3DParams) -> np.ndarray:
    """
    3D Gaussian "defragmentation" kernel as a potential profile on the lattice.

    We build a potential V_defrag(r) that is largest near the center and then
    use it in an effective nonlocal term (implemented in position space in this
    finite model).
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz

    x0 = 0.5 * (Lx - 1)
    y0 = 0.5 * (Ly - 1)
    z0 = 0.5 * (Lz - 1)

    sigma2 = params.sigma_defrag ** 2
    V = np.zeros(Ns, dtype=float)

    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                r = site_index_3d(x, y, z, Lx, Ly, Lz)
                dx = x - x0
                dy = y - y0
                dz = z - z0
                rr = dx * dx + dy * dy + dz * dz
                V[r] = np.exp(-rr / (2.0 * sigma2)) if sigma2 > 0 else 0.0

    s = V.sum()
    if s > 0.0:
        V /= s
    return V


# =============================================================================
# Hamiltonian construction
# =============================================================================


def build_twofermion3d_hamiltonian(params: TwoFermion3DParams) -> sp.csr_matrix:
    """
    Construct the sparse Hamiltonian for two spin-1/2 excitations on the 3D lattice.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    J_hop = params.J_hop
    m_eff = params.mass

    g_defrag = params.g_defrag
    lambda_G = params.lambda_G
    lambda_S = params.lambda_S
    lambda_T = params.lambda_T
    J_exch = params.J_exch

    neighbors = build_neighbors_3d(Lx, Ly, Lz)
    V_defrag = defrag_potential_3d(params)

    rows = []
    cols = []
    data = []

    def add(i: int, j: int, val: complex) -> None:
        rows.append(i)
        cols.append(j)
        data.append(val)

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)

        # Onsite mass for each particle
        add(idx, idx, m_eff)

        # Hopping for particle 1
        for r1n in neighbors[r1]:
            j = encode_basis_3d(r1n, s1, r2, s2, Ns)
            add(idx, j, -J_hop)

        # Hopping for particle 2
        for r2n in neighbors[r2]:
            j = encode_basis_3d(r1, s1, r2n, s2, Ns)
            add(idx, j, -J_hop)

        # Defrag potential: attraction to central region
        V_pair = V_defrag[r1] + V_defrag[r2]
        add(idx, idx, g_defrag * V_pair)

        # Gauss-like radial term
        x1, y1, z1 = decode_site_3d(r1, Lx, Ly, Lz)
        x2, y2, z2 = decode_site_3d(r2, Lx, Ly, Lz)

        x0 = 0.5 * (Lx - 1)
        y0 = 0.5 * (Ly - 1)
        z0 = 0.5 * (Lz - 1)

        r1_sq = (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2
        r2_sq = (x2 - x0) ** 2 + (y2 - y0) ** 2 + (z2 - z0) ** 2
        add(idx, idx, lambda_G * (r1_sq + r2_sq))

        # Contact terms at r1==r2: spin interaction, onsite potential, exchange
        if r1 == r2:
            # Heisenberg-like S1·S2 for two spins-1/2
            if s1 == s2:
                S1dotS2 = 0.25
            else:
                S1dotS2 = -0.25
            add(idx, idx, lambda_S * S1dotS2)

            # Onsite "repulsion" / penalty
            add(idx, idx, lambda_T)

            # Explicit exchange term in spin space
            idx_ex = encode_basis_3d(r1, s2, r2, s1, Ns)
            add(idx, idx_ex, J_exch)

    H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    # Ensure Hermiticity
    H = (H + H.getH()) * 0.5
    return H


# =============================================================================
# Diagnostics
# =============================================================================


def antisymmetry_metrics(psi: np.ndarray, params: TwoFermion3DParams) -> Dict[str, float]:
    """
    Exchange antisymmetry diagnostics.

    We work in the basis |i,j> where i and j stand for the combined
    position+spin indices of particle 1 and 2.

    For each pair (i,j) we consider

        psi_ij + psi_ji  (fully symmetric under exchange)
        psi_ij - psi_ji  (fully antisymmetric under exchange)

    and define

        antisym_violation = sum |psi_ij + psi_ji|^2
        sym_violation     = sum |psi_ij - psi_ji|^2

    Over a *normalized* state one finds

        antisym_violation + sym_violation = 4

    and these contributions are proportional to the weights in the
    symmetric / antisymmetric subspaces.  We therefore define:

        w_A = sym_violation     / (antisym_violation + sym_violation)
        w_S = antisym_violation / (antisym_violation + sym_violation)

    and report

        antisym_score = w_A
        sym_score     = w_S

    so that antisym_score, sym_score ∈ [0,1] and sum to 1.  A perfectly
    antisymmetric state has antisym_score=1; a perfectly symmetric
    state has antisym_score=0.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    total_norm = float(np.vdot(psi, psi).real)
    if total_norm <= 0.0:
        return {
            "total_norm": total_norm,
            "antisym_violation": 0.0,
            "sym_violation": 0.0,
            "antisym_score": 0.0,
            "sym_score": 0.0,
        }

    psi = psi / np.sqrt(total_norm)
    total_norm = 1.0

    antisym_violation = 0.0
    sym_violation = 0.0

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        idx_ex = encode_basis_3d(r2, s2, r1, s1, Ns)

        psi_ij = psi[idx]
        psi_ji = psi[idx_ex]

        antisym_violation += abs(psi_ij + psi_ji) ** 2
        sym_violation     += abs(psi_ij - psi_ji) ** 2

    total_vs = antisym_violation + sym_violation
    if total_vs <= 0.0:
        antisym_score = 0.0
        sym_score = 0.0
    else:
        sym_score = float(antisym_violation / total_vs)
        antisym_score = float(sym_violation / total_vs)

    return {
        "total_norm": total_norm,
        "antisym_violation": float(antisym_violation),
        "sym_violation": float(sym_violation),
        "antisym_score": float(antisym_score),
        "sym_score": float(sym_score),
    }


def overlap_and_spin_metrics(psi: np.ndarray, params: TwoFermion3DParams) -> Dict[str, float]:
    """
    Overlap (r1 == r2) and spin decomposition:

    - overlap_prob          : probability r1 == r2
    - singlet_same_site     : total singlet weight at overlap
    - triplet_same_site     : total triplet weight at overlap
    - singlet_fraction      : singlet_same_site / (singlet+triplet) in the overlap region
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))

    overlap_prob = 0.0
    singlet_same_site = 0.0
    triplet_same_site = 0.0

    for r in range(Ns):
        # basis order: s ∈ {0 (↑), 1 (↓)}
        amp_uu = psi[encode_basis_3d(r, 0, r, 0, Ns)]
        amp_ud = psi[encode_basis_3d(r, 0, r, 1, Ns)]
        amp_du = psi[encode_basis_3d(r, 1, r, 0, Ns)]
        amp_dd = psi[encode_basis_3d(r, 1, r, 1, Ns)]

        site_prob = (
            abs(amp_uu) ** 2
            + abs(amp_ud) ** 2
            + abs(amp_du) ** 2
            + abs(amp_dd) ** 2
        )
        overlap_prob += site_prob

        # Proper singlet / triplet decomposition for the two spins at this site
        sing_amp = (amp_ud - amp_du) / np.sqrt(2.0)
        trip_m1 = amp_uu
        trip_m0 = (amp_ud + amp_du) / np.sqrt(2.0)
        trip_m_1 = amp_dd

        singlet_same_site += abs(sing_amp) ** 2
        triplet_same_site += (
            abs(trip_m1) ** 2 + abs(trip_m0) ** 2 + abs(trip_m_1) ** 2
        )

    total_overlap_spin = singlet_same_site + triplet_same_site
    if total_overlap_spin > 1e-12:
        singlet_fraction = singlet_same_site / total_overlap_spin
    else:
        singlet_fraction = 0.0

    return {
        "overlap_prob": float(overlap_prob),
        "singlet_same_site": float(singlet_same_site),
        "triplet_same_site": float(triplet_same_site),
        "singlet_fraction": float(singlet_fraction),
    }


def gauss_energy_expectation(psi: np.ndarray, params: TwoFermion3DParams) -> float:
    """
    Expectation value of the Gauss-like radial operator:
      lambda_G * (r1^2 + r2^2).
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    total_norm = float(np.vdot(psi, psi).real)
    if total_norm <= 0.0:
        return 0.0

    psi = psi / np.sqrt(total_norm)

    x0 = 0.5 * (Lx - 1)
    y0 = 0.5 * (Ly - 1)
    z0 = 0.5 * (Lz - 1)

    E_gauss = 0.0
    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        amp = psi[idx]
        p = abs(amp) ** 2

        x1, y1, z1 = decode_site_3d(r1, Lx, Ly, Lz)
        x2, y2, z2 = decode_site_3d(r2, Lx, Ly, Lz)

        r1_sq = (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2
        r2_sq = (x2 - x0) ** 2 + (y2 - y0) ** 2 + (z2 - z0) ** 2

        E_gauss += params.lambda_G * (r1_sq + r2_sq) * p

    return float(E_gauss)


def compute_lump_densities(psi: np.ndarray, params: TwoFermion3DParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute single-particle densities rho1(r), rho2(r) and rho_tot(r) = rho1+rho2.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    total_norm = float(np.vdot(psi, psi).real)
    if total_norm <= 0.0:
        return np.zeros(Ns), np.zeros(Ns), np.zeros(Ns)

    psi = psi / np.sqrt(total_norm)

    rho1 = np.zeros(Ns, dtype=float)
    rho2 = np.zeros(Ns, dtype=float)

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        amp = psi[idx]
        p = abs(amp) ** 2

        rho1[r1] += p
        rho2[r2] += p

    rho_tot = rho1 + rho2
    return rho1, rho2, rho_tot


# =============================================================================
# Spin reduced state and CHSH
# =============================================================================


def reduced_spin_density_matrix(psi: np.ndarray, params: TwoFermion3DParams) -> np.ndarray:
    """
    Compute the reduced 4x4 spin density matrix for the two spins (s1,s2),
    tracing out positions r1,r2.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    total_norm = float(np.vdot(psi, psi).real)
    if total_norm <= 0.0:
        return np.zeros((4, 4), dtype=complex)

    psi = psi / np.sqrt(total_norm)

    rho_spin = np.zeros((4, 4), dtype=complex)

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        amp = psi[idx]
        if amp == 0:
            continue

        for jdx in range(dim):
            r1p, s1p, r2p, s2p = decode_basis_3d(jdx, Ns)
            if r1p != r1 or r2p != r2:
                continue

            amp_p = psi[jdx]
            bra_s = 2 * s1 + s2
            ket_s = 2 * s1p + s2p
            rho_spin[bra_s, ket_s] += amp * np.conjugate(amp_p)

    # Normalize trace to 1
    tr = np.trace(rho_spin)
    if abs(tr) > 0.0:
        rho_spin /= tr

    return rho_spin


def chsh_S_from_rho(rho_spin: np.ndarray) -> float:
    """
    Given a 4x4 2-qubit density matrix, compute the maximum CHSH S-value
    via the Horodecki formula.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    paulis = [sx, sy, sz]

    T = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            op = np.kron(paulis[i], paulis[j])
            val = np.trace(rho_spin @ op)
            T[i, j] = float(np.real(val))

    TT = T.T @ T
    eigvals, _ = np.linalg.eigh(TT)
    eigvals = np.sort(eigvals)[::-1]
    m1, m2 = eigvals[0], eigvals[1]
    S_max = 2.0 * np.sqrt(max(m1 + m2, 0.0))
    return float(S_max)


def chsh_S_from_state(psi: np.ndarray, params: TwoFermion3DParams) -> float:
    """
    Convenience wrapper: build reduced spin density matrix from |psi> and
    compute its CHSH S-value.
    """
    rho_spin = reduced_spin_density_matrix(psi, params)
    return chsh_S_from_rho(rho_spin)


# =============================================================================
# Top-level experiment driver
# =============================================================================


def run_twofermion3d_experiment(params: TwoFermion3DParams) -> Dict[str, Any]:
    print("======================================================================")
    print("3D TWO-EXCITATION SUBSTRATE (fermion-like model)")
    print("======================================================================")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("----------------------------------------------------------------------")

    H = build_twofermion3d_hamiltonian(params)
    dim = H.shape[0]
    print(f"[INFO] Hilbert dimension (two excitations) = {dim}")
    print("[INFO] Solving for ground state (smallest eigenvalue) with eigsh...")
    evals, evecs = eigsh(H, k=params.k_eigs, which="SA", maxiter=params.max_eigsh_iter)
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    print(f"[RESULT] Ground state energy E0 = {E0:.6f}")
    print("----------------------------------------------------------------------")

    anti = antisymmetry_metrics(psi0, params)
    print("Exchange antisymmetry diagnostics:")
    print(f"  Antisymmetry score A = {anti['antisym_score']:.6f}")
    print(f"  Symmetry score S     = {anti['sym_score']:.6f}")
    print(f"  Antisym violation    = {anti['antisym_violation']:.6e}")
    print()

    overlap = overlap_and_spin_metrics(psi0, params)
    print("Overlap & spin diagnostics (r1 == r2):")
    print(f"  Spatial overlap prob (r1 == r2)     = {overlap['overlap_prob']:.6f}")
    print(f"  Singlet weight at overlap           = {overlap['singlet_same_site']:.6f}")
    print(f"  Triplet weight at overlap           = {overlap['triplet_same_site']:.6f}")
    print(f"  Singlet fraction (overlap region)   = {overlap['singlet_fraction']:.6f}")
    print()

    E_gauss = gauss_energy_expectation(psi0, params)
    print("Gauss-like energy (expectation):")
    print(f"  <H_gauss> = {E_gauss:.6f}")
    print()

    rho1, rho2, rho_tot = compute_lump_densities(psi0, params)
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz

    print("Lump diagnostics (3D site densities):")
    for r in range(Ns):
        x, y, z = decode_site_3d(r, Lx, Ly, Lz)
        print(
            f"  Site (x={x}, y={y}, z={z}): "
            f"rho1={rho1[r]:.4f}, rho2={rho2[r]:.4f}, rho_tot={rho_tot[r]:.4f}"
        )
    print()

    S_chsh = chsh_S_from_state(psi0, params)
    print("CHSH diagnostics (spin sector, reduced density matrix):")
    print(f"  S = {S_chsh:.6f}, |S| = {abs(S_chsh):.6f}")
    print("  (|S| <= 2: local realistic, 2 < |S| <= 2√2: quantum-allowed)")
    print("----------------------------------------------------------------------")

    if anti["antisym_score"] > 0.95 and overlap["singlet_fraction"] > 0.95:
        print("[VERDICT] Ground state is strongly exchange-antisymmetric with "
              "purely singlet overlap (fermion-like) in this 3D toy model.")
    else:
        print("[VERDICT] Antisymmetry and/or singlet preference is partial or "
              "absent in this parameter regime.")
    if abs(S_chsh) > 2.0:
        print("[VERDICT] Spin sector exhibits CHSH violation (Bell-inequality "
              "violation) in this reduced 2-qubit state.")
    else:
        print("[VERDICT] No CHSH violation in the spin sector for these parameters.")
    print("======================================================================")

    return {
        "E0": E0,
        "antisymmetry": anti,
        "overlap": overlap,
        "E_gauss": E_gauss,
        "rho1": rho1,
        "rho2": rho2,
        "rho_tot": rho_tot,
        "S_chsh": S_chsh,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="3D two-excitation substrate model with fermion-like "
                    "antisymmetry and CHSH diagnostics."
    )
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--Lz", type=int, default=2)

    p.add_argument("--J-hop", type=float, default=1.0, dest="J_hop")
    p.add_argument("--mass", type=float, default=0.1)

    p.add_argument("--g-defrag", type=float, default=1.0)
    p.add_argument("--sigma-defrag", type=float, default=1.0)

    p.add_argument("--lambda-G", type=float, default=5.0, dest="lambda_G")
    p.add_argument("--lambda-S", type=float, default=-1.0, dest="lambda_S")
    p.add_argument("--lambda-T", type=float, default=0.0, dest="lambda_T")
    p.add_argument("--J-exch", type=float, default=1.0)

    p.add_argument("--max-eigsh-iter", type=int, default=5000)
    p.add_argument("--k-eigs", type=int, default=1)

    args = p.parse_args()

    params = TwoFermion3DParams(
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        J_hop=args.J_hop,
        mass=args.mass,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        lambda_G=args.lambda_G,
        lambda_S=args.lambda_S,
        lambda_T=args.lambda_T,
        J_exch=args.J_exch,
        max_eigsh_iter=args.max_eigsh_iter,
        k_eigs=args.k_eigs,
    )

    run_twofermion3d_experiment(params)


if __name__ == "__main__":
    main()
