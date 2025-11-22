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
  - H_mass     : mass term m per excitation.
  - H_defrag   : "defrag" potential favoring central clumping in 3D.
  - H_Gauss    : Gauss-like occupancy penalty on local deviations from a
                 target occupancy rho0 = 2 / Ns.
  - H_contact  : local spin interaction at r1 == r2:
                   * Heisenberg S1·S2 with strength J_exch,
                   * Singlet bonus lambda_S,
                   * Triplet penalty lambda_T.

We:
  1) Build H, diagonalize its ground state with eigsh.
  2) Compute:
       - exchange antisymmetry diagnostics,
       - overlap probability and spin (singlet/triplet) at overlap,
       - Gauss-like energy expectation,
       - 3D "lump" densities for each excitation and total.
  3) Compute a CHSH S from the reduced 2-spin density matrix obtained by
     tracing out positions.

This is a 3D generalization of the 2D two-excitation substrate model,
and it provides:
  - fermion-like antisymmetry (if the ground state selects that sector),
  - real-space lump structure in 3D,
  - a Bell/CHSH test on the same Hilbert space.
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

from qutip import Qobj, tensor, sigmax, sigmay, sigmaz, qeye


# =============================================================================
# Dataclass for parameters
# =============================================================================

@dataclass
class TwoFermion3DParams:
    Lx: int = 2
    Ly: int = 2
    Lz: int = 2

    J_hop: float = 1.0
    m: float = 0.1

    g_defrag: float = 1.0
    sigma_defrag: float = 1.0

    lambda_G: float = 5.0
    lambda_S: float = -1.0
    lambda_T: float = 0.0
    J_exch: float = 1.0

    max_eigsh_iter: int = 5000
    k_eigs: int = 1  # number of eigenvalues to find (ground state only)


# =============================================================================
# Lattice helpers (3D)
# =============================================================================

def site_index_3d(x: int, y: int, z: int, Lx: int, Ly: int, Lz: int) -> int:
    return x + Lx * (y + Ly * z)


def site_coords_3d(r: int, Lx: int, Ly: int, Lz: int) -> Tuple[int, int, int]:
    x = r % Lx
    tmp = r // Lx
    y = tmp % Ly
    z = tmp // Ly
    return x, y, z


def build_neighbors_3d(Lx: int, Ly: int, Lz: int) -> Dict[int, list]:
    """
    3D nearest-neighbor list with periodic boundary conditions.
    """
    Ns = Lx * Ly * Lz
    neighbors = {r: [] for r in range(Ns)}
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
        dist2 = dx * dx + dy * dy + dz * dz
        V[r] = -np.exp(-dist2 / (2.0 * sigma * sigma))

    return V


# =============================================================================
# Basis mapping: |r1,s1; r2,s2> <-> index
# =============================================================================

def encode_basis_3d(r1: int, s1: int, r2: int, s2: int, Ns: int) -> int:
    """
    s1,s2 in {0,1} for (↑,↓).
    index ranges 0..dim-1, where dim = (Ns*2)*(Ns*2).
    """
    return (((r1 * 2 + s1) * Ns * 2) + (r2 * 2 + s2))


def decode_basis_3d(idx: int, Ns: int) -> Tuple[int, int, int, int]:
    """
    Inverse mapping of encode_basis_3d.
    """
    tmp = idx
    r2s2 = tmp % (Ns * 2)
    tmp //= (Ns * 2)
    r1s1 = tmp

    r1 = r1s1 // 2
    s1 = r1s1 % 2
    r2 = r2s2 // 2
    s2 = r2s2 % 2
    return r1, s1, r2, s2


# =============================================================================
# Hamiltonian construction
# =============================================================================

def build_twofermion3d_hamiltonian(params: TwoFermion3DParams) -> csr_matrix:
    """
    Build the 3D two-excitation Hamiltonian:

        H = H_hop + H_mass + H_defrag + H_Gauss + H_contact

    as a sparse CSR matrix.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    neighbors = build_neighbors_3d(Lx, Ly, Lz)
    V_defrag = defrag_potential_3d(params)
    rho0 = 2.0 / Ns   # target occupancy per site for Gauss penalty

    H = lil_matrix((dim, dim), dtype=np.complex128)

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)

        # 1) mass: m per excitation
        H[idx, idx] += 2.0 * params.m

        # 2) defrag: g_defrag * (V_defrag[r1] + V_defrag[r2])
        H[idx, idx] += params.g_defrag * (V_defrag[r1] + V_defrag[r2])

        # 3) Gauss-like occupancy penalty
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        G = occ.astype(float) - rho0
        gauss_energy = 0.5 * params.lambda_G * np.sum(G * G)
        H[idx, idx] += gauss_energy

        # 4) Contact spin term at overlap: r1 == r2
        if r1 == r2:
            # S^z = +1/2 for s=0 (↑), -1/2 for s=1 (↓)
            sz1 = +0.5 if s1 == 0 else -0.5
            sz2 = +0.5 if s2 == 0 else -0.5

            # Heisenberg diagonal piece: S1·S2 ~ Sz1*Sz2
            H[idx, idx] += params.J_exch * (sz1 * sz2)

            # Triplet penalty for equal spins
            if s1 == s2:
                H[idx, idx] += params.lambda_T

            # if opposite spins, add singlet bonus + spin-flip mixing
            if s1 != s2:
                # Singlet bonus
                H[idx, idx] += params.lambda_S

                # flip spins: |↑↓> <-> |↓↑>
                s1p, s2p = s2, s1
                idx_flip = encode_basis_3d(r1, s1p, r2, s2p, Ns)
                H[idx_flip, idx] += 0.5 * params.J_exch

        # 5) Hopping (particle 1)
        for r1p in neighbors[r1]:
            idx_new = encode_basis_3d(r1p, s1, r2, s2, Ns)
            H[idx_new, idx] += -params.J_hop

        # 6) Hopping (particle 2)
        for r2p in neighbors[r2]:
            idx_new = encode_basis_3d(r1, s1, r2p, s2, Ns)
            H[idx_new, idx] += -params.J_hop

    return H.tocsr()


# =============================================================================
# Diagnostics: antisymmetry, overlap, Gauss energy, lump densities
# =============================================================================

def antisymmetry_metrics(psi: np.ndarray, params: TwoFermion3DParams) -> Dict[str, float]:
    """
    Exchange antisymmetry diagnostics:

    - antisym_violation = sum |psi(i,j) + psi(j,i)|^2
    - sym_violation     = sum |psi(i,j) - psi(j,i)|^2
    - antisym_score     = 1 - antisym_violation / norm
    - sym_score         = 1 - sym_violation / norm

    where "i" and "j" stand for the pair (r1,s1) and (r2,s2).
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    total_norm = float(np.vdot(psi, psi).real)

    antisym_violation = 0.0
    sym_violation = 0.0

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        idx_ex = encode_basis_3d(r2, s2, r1, s1, Ns)

        psi_ij = psi[idx]
        psi_ji = psi[idx_ex]

        antisym_violation += abs(psi_ij + psi_ji) ** 2
        sym_violation     += abs(psi_ij - psi_ji) ** 2

    antisym_score = 1.0 - antisym_violation / total_norm
    sym_score     = 1.0 - sym_violation / total_norm

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
    - singlet_fraction      : singlet_same_site / (singlet+triplet)
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))

    overlap_prob = 0.0
    singlet_same_site = 0.0
    triplet_same_site = 0.0

    for r in range(Ns):
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

        # singlet / triplet
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
    Expectation value of Gauss-like penalty term.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    rho0 = 2.0 / Ns

    E_gauss = 0.0
    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) < 1e-14:
            continue

        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        G = occ.astype(float) - rho0
        gauss_energy = 0.5 * params.lambda_G * np.sum(G * G)
        E_gauss += gauss_energy * abs(amp) ** 2

    return float(E_gauss)


def compute_lump_densities(psi: np.ndarray, params: TwoFermion3DParams):
    """
    Compute marginal densities rho1(r), rho2(r), and total rho(r) in 3D.

    rho1(r) = sum_{s1,s2,r2} |psi(r,s1; r2,s2)|^2
    rho2(r) = sum_{s1,s2,r1} |psi(r1,s1; r,s2)|^2
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))

    rho1 = np.zeros(Ns, dtype=float)
    rho2 = np.zeros(Ns, dtype=float)

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        amp2 = abs(psi[idx]) ** 2
        rho1[r1] += amp2
        rho2[r2] += amp2

    rho_tot = rho1 + rho2

    return rho1, rho2, rho_tot


# =============================================================================
# CHSH from reduced spin density matrix
# =============================================================================

def reduced_spin_density_matrix(psi: np.ndarray, params: TwoFermion3DParams) -> Qobj:
    """
    Trace out positions, leaving a 2-qubit spin density matrix:

        rho_spin[s1,s2; s1',s2'] = sum_{r1,r2} psi(r1,s1; r2,s2) psi*(r1,s1'; r2,s2').
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))

    rho_spin = np.zeros((4, 4), dtype=complex)

    for r1 in range(Ns):
        for r2 in range(Ns):
            for s1 in (0, 1):
                for s2 in (0, 1):
                    idx = encode_basis_3d(r1, s1, r2, s2, Ns)
                    amp = psi[idx]
                    if abs(amp) < 1e-14:
                        continue
                    for s1p in (0, 1):
                        for s2p in (0, 1):
                            idxp = encode_basis_3d(r1, s1p, r2, s2p, Ns)
                            ampp = psi[idxp]
                            bra_idx = 2 * s1 + s2
                            ket_idx = 2 * s1p + s2p
                            rho_spin[bra_idx, ket_idx] += amp * np.conjugate(ampp)

    # normalize (should already be normalized if psi is)
    tr = np.trace(rho_spin)
    if abs(tr) > 1e-14:
        rho_spin /= tr

    # explicitly declare as 2-qubit density matrix
    return Qobj(rho_spin, dims=[[2, 2], [2, 2]])


def build_chsh_operator() -> Qobj:
    """
    Build the standard CHSH operator:

        B = A⊗B + A⊗B' + A'⊗B - A'⊗B'

    with settings:
        A  = σ_z
        A' = σ_x
        B  = (σ_z + σ_x)/√2
        B' = (σ_z - σ_x)/√2
    which give Tsirelson bound |S| = 2√2 for the singlet.
    """
    sx = sigmax()
    sz = sigmaz()
    I = qeye(2)

    A = sz
    Ap = sx
    B = (sz + sx) / np.sqrt(2.0)
    Bp = (sz - sx) / np.sqrt(2.0)

    A1 = tensor(A, I)
    Ap1 = tensor(Ap, I)
    B2 = tensor(I, B)
    Bp2 = tensor(I, Bp)

    B_chsh = A1 * B2 + A1 * Bp2 + Ap1 * B2 - Ap1 * Bp2
    return B_chsh


def chsh_S_from_state(psi: np.ndarray, params: TwoFermion3DParams) -> float:
    """
    Compute CHSH S from the 2-spin reduced density matrix of psi:

        S = Tr( rho_spin * B_CHSH )
    """
    rho_spin = reduced_spin_density_matrix(psi, params)  # 2-qubit density matrix
    B_chsh = build_chsh_operator()                      # 2-qubit operator
    S = (rho_spin * B_chsh).tr()
    return float(np.real(S))


# =============================================================================
# Main experiment
# =============================================================================

def run_twofermion3d_experiment(params: TwoFermion3DParams) -> Dict[str, Any]:
    print("======================================================================")
    print("3D TWO-EXCITATION SUBSTRATE (fermion-like model)")
    print("======================================================================")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("----------------------------------------------------------------------")

    # Build Hamiltonian
    H = build_twofermion3d_hamiltonian(params)
    dim = H.shape[0]
    print(f"[INFO] Hilbert dimension (two excitations) = {dim}")
    print("[INFO] Solving for ground state (smallest eigenvalue) with eigsh...")
    evals, evecs = eigsh(H, k=params.k_eigs, which="SA", maxiter=params.max_eigsh_iter)
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    # Normalize
    norm = np.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    print(f"[RESULT] Ground state energy E0 = {E0:.6f}")
    print("----------------------------------------------------------------------")

    # Antisymmetry
    anti = antisymmetry_metrics(psi0, params)
    print("Exchange antisymmetry diagnostics:")
    print(f"  Antisymmetry score A = {anti['antisym_score']:.6f}")
    print(f"  Symmetry score S     = {anti['sym_score']:.6f}")
    print(f"  Antisym violation    = {anti['antisym_violation']:.6e}")
    print()

    # Overlap + spin
    overlap = overlap_and_spin_metrics(psi0, params)
    print("Overlap & spin diagnostics (r1 == r2):")
    print(f"  Spatial overlap prob (r1 == r2)     = {overlap['overlap_prob']:.6f}")
    print(f"  Singlet weight at overlap           = {overlap['singlet_same_site']:.6f}")
    print(f"  Triplet weight at overlap           = {overlap['triplet_same_site']:.6f}")
    print(f"  Singlet fraction (overlap region)   = {overlap['singlet_fraction']:.6f}")
    print()

    # Gauss energy
    E_gauss = gauss_energy_expectation(psi0, params)
    print("Gauss-like energy (expectation):")
    print(f"  <H_gauss> = {E_gauss:.6f}")
    print()

    # Lump densities
    rho1, rho2, rho_tot = compute_lump_densities(psi0, params)
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz

    print("Lump diagnostics (3D site densities):")
    for r in range(Ns):
        x, y, z = site_coords_3d(r, Lx, Ly, Lz)
        print(f"  Site (x={x}, y={y}, z={z}): "
              f"rho1={rho1[r]:.4f}, rho2={rho2[r]:.4f}, rho_tot={rho_tot[r]:.4f}")
    print()

    # CHSH from reduced spin state
    S_chsh = chsh_S_from_state(psi0, params)
    print("CHSH diagnostics (spin sector, reduced density matrix):")
    print(f"  S = {S_chsh:.6f}, |S| = {abs(S_chsh):.6f}")
    print("  (|S| <= 2: local realistic, 2 < |S| <= 2√2: quantum-allowed)")
    print("----------------------------------------------------------------------")

    # Verdict
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

def parse_args():
    p = argparse.ArgumentParser(
        description="3D two-excitation finite-Hilbert substrate with "
                    "fermion-like antisymmetry and CHSH diagnostics."
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
    p.add_argument("--J-exch", type=float, default=1.0, dest="J_exch")

    p.add_argument("--max-eigsh-iter", type=int, default=5000)
    p.add_argument("--k-eigs", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()

    params = TwoFermion3DParams(
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        J_hop=args.J_hop,
        m=args.mass,
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
