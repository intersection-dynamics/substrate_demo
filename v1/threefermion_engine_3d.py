#!/usr/bin/env python3
"""
threefermion_engine_3d.py

3D finite-Hilbert substrate with THREE spin-1/2 excitations on a 3D lattice.

- Lattice: Lx x Ly x Lz sites, Ns = Lx*Ly*Lz.
- Degrees of freedom: positions r1, r2, r3 (sites), spins s1, s2, s3 (↑/↓).
- Basis states: |r1, s1; r2, s2; r3, s3>, with r in {0,...,Ns-1}, s in {0,1}.
- Hilbert dimension: dim = (Ns * 2)^3 = (2*Ns)^3.

Hamiltonian (three-excitation sector):
    H = H_hop + H_mass + H_defrag + H_Gauss + H_contact

where:
  - H_hop      : nearest-neighbor hopping for each excitation (3D periodic BC).
  - H_mass     : mass term m per excitation.
  - H_defrag   : "defrag" potential, a 3D Gaussian well centered in the box.
  - H_Gauss    : Gauss-like occupancy penalty for deviations from uniform density.
  - H_contact  : spin/contact interactions when two or more excitations coincide
                 on the same site, built as a sum over pairs (1–2, 1–3, 2–3):

        * Heisenberg-like diagonal term J_exch * (Sz_i * Sz_j)
        * Triplet penalty lambda_T for equal spins
        * Singlet bonus lambda_S for opposite spins
        * Spin-flip mixing (|↑↓> <-> |↓↑>) with amplitude 0.5 * J_exch

Diagnostics provided:
  - Gauss energy expectation
  - Double / triple overlap probabilities
  - Site densities (rho1, rho2, rho3, rho_tot)
  - S3 antisymmetry weight, pairwise antisymmetry
  - 3-spin reduced density matrix
  - Pairwise spin RDMs (rho12, rho13, rho23)
  - Pair singlet fractions & CHSH S_ij
  - Spin sector weights (S_tot = 1/2 vs 3/2)
  - Simple "2+1" fermion-like classification and verdict
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import functools
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


# =============================================================================
# Dataclass for parameters
# =============================================================================


@dataclass
class ThreeFermion3DParams:
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


def defrag_potential_3d(params: ThreeFermion3DParams) -> np.ndarray:
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
# Basis mapping: |r1,s1; r2,s2; r3,s3> <-> index
# =============================================================================


def encode_basis_3body(
    r1: int, s1: int, r2: int, s2: int, r3: int, s3: int, Ns: int
) -> int:
    """
    3-body basis encoding.

    s1,s2,s3 in {0,1} for (↑,↓).
    index ranges 0..dim-1, where dim = (Ns*2)^3.
    """
    dim_1 = Ns * 2
    return (((r1 * 2 + s1) * dim_1 + (r2 * 2 + s2)) * dim_1 + (r3 * 2 + s3))


def decode_basis_3body(idx: int, Ns: int) -> Tuple[int, int, int, int, int, int]:
    """
    Inverse mapping of encode_basis_3body.
    """
    dim_1 = Ns * 2

    tmp = idx
    r3s3 = tmp % dim_1
    tmp //= dim_1
    r2s2 = tmp % dim_1
    tmp //= dim_1
    r1s1 = tmp

    r1 = r1s1 // 2
    s1 = r1s1 % 2
    r2 = r2s2 // 2
    s2 = r2s2 % 2
    r3 = r3s3 // 2
    s3 = r3s3 % 2

    return r1, s1, r2, s2, r3, s3


# =============================================================================
# Hamiltonian construction (3-body)
# =============================================================================


def build_threefermion3d_hamiltonian(params: ThreeFermion3DParams) -> csr_matrix:
    """
    Build the 3D three-excitation Hamiltonian:

        H = H_hop + H_mass + H_defrag + H_Gauss + H_contact

    as a sparse CSR matrix.

    The contact term H_contact is built as a sum over pairs of excitations
    (1–2, 1–3, 2–3). When two excitations share the same site, we apply:

      - Heisenberg-like diagonal term:  J_exch * (Sz_i * Sz_j)
      - Triplet penalty (equal spins): lambda_T
      - Singlet bonus (opposite spins): lambda_S
      - Spin-flip mixing: |↑↓> <-> |↓↑> with amplitude 0.5 * J_exch

    These contributions accumulate if three excitations occupy the same site
    (i.e., all three pairs overlap).
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim_1 = Ns * 2
    dim = dim_1 * dim_1 * dim_1

    neighbors = build_neighbors_3d(Lx, Ly, Lz)
    V_defrag = defrag_potential_3d(params)
    rho0 = 3.0 / Ns   # target occupancy per site for Gauss penalty (3 excitations)

    H = lil_matrix((dim, dim), dtype=np.complex128)

    for idx in range(dim):
        r1, s1, r2, s2, r3, s3 = decode_basis_3body(idx, Ns)

        # 1) mass: m per excitation
        H[idx, idx] += 3.0 * params.m

        # 2) defrag: g_defrag * (V_defrag[r1] + V_defrag[r2] + V_defrag[r3])
        H[idx, idx] += params.g_defrag * (
            V_defrag[r1] + V_defrag[r2] + V_defrag[r3]
        )

        # 3) Gauss-like occupancy penalty
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        occ[r3] += 1
        G = occ.astype(float) - rho0
        gauss_energy = 0.5 * params.lambda_G * np.sum(G * G)
        H[idx, idx] += gauss_energy

        # 4) Contact spin term when sites coincide
        def add_pair_contact(
            r_i: int,
            s_i: int,
            r_j: int,
            s_j: int,
            r_k: int,
            s_k: int,
            which_pair: str,
        ):
            """
            Apply contact terms for a specific pair (i,j) if r_i == r_j:

              - Heisenberg Sz_i Sz_j
              - triplet penalty
              - singlet bonus
              - spin-flip mixing for opposite spins
            """
            nonlocal H, idx

            if r_i != r_j:
                return

            # S^z = +1/2 for s=0 (↑), -1/2 for s=1 (↓)
            sz_i = +0.5 if s_i == 0 else -0.5
            sz_j = +0.5 if s_j == 0 else -0.5

            # Heisenberg diagonal piece: S_i·S_j ~ Sz_i*Sz_j
            H[idx, idx] += params.J_exch * (sz_i * sz_j)

            # Triplet penalty for equal spins
            if s_i == s_j:
                H[idx, idx] += params.lambda_T

            # Singlet bonus + spin flip if opposite spins
            if s_i != s_j:
                H[idx, idx] += params.lambda_S

                # flip spins: |↑↓> <-> |↓↑> for the chosen pair
                if which_pair == "12":
                    s1p, s2p = s_j, s_i
                    s3p = s_k
                    r1p, r2p, r3p = r_i, r_j, r_k
                elif which_pair == "13":
                    s1p, s3p = s_j, s_i
                    s2p = s_k
                    r1p, r2p, r3p = r_i, r_k, r_j
                elif which_pair == "23":
                    s2p, s3p = s_j, s_i
                    s1p = s_k
                    r1p, r2p, r3p = r_k, r_i, r_j
                else:
                    return

                idx_flip = encode_basis_3body(
                    r1p, s1p, r2p, s2p, r3p, s3p, Ns
                )
                H[idx_flip, idx] += 0.5 * params.J_exch

        # Apply contact for each pair if overlapping
        add_pair_contact(r1, s1, r2, s2, r3, s3, "12")
        add_pair_contact(r1, s1, r3, s3, r2, s2, "13")
        add_pair_contact(r2, s2, r3, s3, r1, s1, "23")

        # 5) Hopping (particle 1)
        for r1p in neighbors[r1]:
            idx_new = encode_basis_3body(r1p, s1, r2, s2, r3, s3, Ns)
            H[idx_new, idx] += -params.J_hop

        # 6) Hopping (particle 2)
        for r2p in neighbors[r2]:
            idx_new = encode_basis_3body(r1, s1, r2p, s2, r3, s3, Ns)
            H[idx_new, idx] += -params.J_hop

        # 7) Hopping (particle 3)
        for r3p in neighbors[r3]:
            idx_new = encode_basis_3body(r1, s1, r2, s2, r3p, s3, Ns)
            H[idx_new, idx] += -params.J_hop

    return H.tocsr()


# =============================================================================
# Diagnostics: Gauss energy, overlaps, site densities
# =============================================================================


def gauss_energy_expectation_3body(
    psi: np.ndarray, params: ThreeFermion3DParams
) -> float:
    """
    Expectation value of Gauss-like penalty term for 3 excitations.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim_1 = Ns * 2
    dim = dim_1 * dim_1 * dim_1

    psi = psi.reshape((dim,))
    rho0 = 3.0 / Ns

    E_gauss = 0.0
    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) < 1e-14:
            continue

        r1, s1, r2, s2, r3, s3 = decode_basis_3body(idx, Ns)
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        occ[r3] += 1
        G = occ.astype(float) - rho0
        gauss_energy = 0.5 * params.lambda_G * np.sum(G * G)

        E_gauss += gauss_energy * (abs(amp) ** 2)

    return float(E_gauss)


def overlap_metrics_3body(
    psi: np.ndarray, params: ThreeFermion3DParams
) -> Dict[str, float]:
    """
    Compute double and triple overlap probabilities:

      - overlap_prob_double : probability that at least two excitations
                              occupy the same site (r_i == r_j for some i<j)
      - overlap_prob_triple : probability that all three excitations occupy
                              the same site (r1 == r2 == r3)
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim_1 = Ns * 2
    dim = dim_1 * dim_1 * dim_1

    psi = psi.reshape((dim,))

    overlap_double = 0.0
    overlap_triple = 0.0

    for idx in range(dim):
        amp2 = abs(psi[idx]) ** 2
        if amp2 < 1e-16:
            continue

        r1, s1, r2, s2, r3, s3 = decode_basis_3body(idx, Ns)

        # triple overlap
        if (r1 == r2) and (r2 == r3):
            overlap_triple += amp2
            overlap_double += amp2  # triple also counts as double
        else:
            # double overlap: at least one pair coincides
            if (r1 == r2) or (r1 == r3) or (r2 == r3):
                overlap_double += amp2

    return {
        "overlap_prob_double": float(overlap_double),
        "overlap_prob_triple": float(overlap_triple),
    }


def site_densities_3body(
    psi: np.ndarray, params: ThreeFermion3DParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one-body densities for each excitation, plus total density:

      - rho1[r] = probability that excitation 1 is at site r
      - rho2[r] = same for excitation 2
      - rho3[r] = same for excitation 3
      - rho_tot[r] = rho1[r] + rho2[r] + rho3[r]

    Each rho_i sums to 1.0; rho_tot sums to 3.0.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim_1 = Ns * 2
    dim = dim_1 * dim_1 * dim_1

    psi = psi.reshape((dim,))

    rho1 = np.zeros(Ns, dtype=float)
    rho2 = np.zeros(Ns, dtype=float)
    rho3 = np.zeros(Ns, dtype=float)

    for idx in range(dim):
        amp2 = abs(psi[idx]) ** 2
        if amp2 < 1e-16:
            continue

        r1, s1, r2, s2, r3, s3 = decode_basis_3body(idx, Ns)
        rho1[r1] += amp2
        rho2[r2] += amp2
        rho3[r3] += amp2

    rho_tot = rho1 + rho2 + rho3
    return rho1, rho2, rho3, rho_tot


# =============================================================================
# Spin-space tools for 3 spins
# =============================================================================


def _spin_triple_index(s1: int, s2: int, s3: int) -> int:
    """
    Map (s1,s2,s3) with s in {0,1} (↑,↓) to 0..7
    using basis ordering: |↑↑↑>,|↑↑↓>,|↑↓↑>,|↑↓↓>,|↓↑↑>,|↓↑↓>,|↓↓↑>,|↓↓↓>.
    """
    return (s1 << 2) | (s2 << 1) | s3


def build_rho_spin_3body(psi: np.ndarray, params: ThreeFermion3DParams) -> np.ndarray:
    """
    Build the 3-spin reduced density matrix rho_spin (8x8) by tracing
    over positions:

        rho_spin[s, s'] = sum_{r1,r2,r3} psi(r,s) psi*(r,s')

    Implemented by accumulating 8-dim spin vectors for each (r1,r2,r3)
    configuration and summing |vec><vec|.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim_1 = Ns * 2
    dim = dim_1 * dim_1 * dim_1

    psi = psi.reshape((dim,))

    spin_blocks: Dict[Tuple[int, int, int], np.ndarray] = {}

    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) < 1e-16:
            continue

        r1, s1, r2, s2, r3, s3 = decode_basis_3body(idx, Ns)
        key = (r1, r2, r3)
        vec = spin_blocks.get(key)
        if vec is None:
            vec = np.zeros(8, dtype=np.complex128)
            spin_blocks[key] = vec
        s_idx = _spin_triple_index(s1, s2, s3)
        vec[s_idx] += amp

    rho_spin = np.zeros((8, 8), dtype=np.complex128)
    for vec in spin_blocks.values():
        rho_spin += np.outer(vec, np.conjugate(vec))

    # Normalize (should already have trace 1, but be safe)
    trace = np.trace(rho_spin)
    if abs(trace) > 1e-14:
        rho_spin /= trace

    return rho_spin


def pair_spin_rdm_from_rho_spin(
    rho_spin: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Given 3-spin RDM rho_spin (8x8), compute the pairwise RDMs rho12, rho13, rho23
    by tracing out the third spin. Spin basis is:

        |s1 s2 s3> with s in {0,1} (↑,↓),

    and we use pair basis |↑↑>,|↑↓>,|↓↑>,|↓↓> with indices 0..3.
    """
    rho12 = np.zeros((4, 4), dtype=np.complex128)
    rho13 = np.zeros((4, 4), dtype=np.complex128)
    rho23 = np.zeros((4, 4), dtype=np.complex128)

    # Loop over spin triples
    for s1 in (0, 1):
        for s2 in (0, 1):
            for s3 in (0, 1):
                a = _spin_triple_index(s1, s2, s3)
                for s1p in (0, 1):
                    for s2p in (0, 1):
                        for s3p in (0, 1):
                            b = _spin_triple_index(s1p, s2p, s3p)
                            val = rho_spin[a, b]
                            if abs(val) < 1e-16:
                                continue

                            # rho12: trace over spin 3
                            if s3 == s3p:
                                i12 = 2 * s1 + s2
                                j12 = 2 * s1p + s2p
                                rho12[i12, j12] += val

                            # rho13: trace over spin 2
                            if s2 == s2p:
                                i13 = 2 * s1 + s3
                                j13 = 2 * s1p + s3p
                                rho13[i13, j13] += val

                            # rho23: trace over spin 1
                            if s1 == s1p:
                                i23 = 2 * s2 + s3
                                j23 = 2 * s2p + s3p
                                rho23[i23, j23] += val

    # Normalize each (should be trace 1)
    for rho in (rho12, rho13, rho23):
        tr = np.trace(rho)
        if abs(tr) > 1e-14:
            rho /= tr

    return {"12": rho12, "13": rho13, "23": rho23}


def singlet_fraction_from_rho2(rho2: np.ndarray) -> float:
    """
    Compute singlet fraction F_S = <S|rho2|S> for two-spin RDM rho2,
    where |S> = (|↑↓> - |↓↑>)/sqrt(2).
    Basis for rho2 is |↑↑>,|↑↓>,|↓↑>,|↓↓>.
    """
    # |↑↓> index 1, |↓↑> index 2
    v = np.zeros(4, dtype=np.complex128)
    v[1] = 1.0 / np.sqrt(2.0)
    v[2] = -1.0 / np.sqrt(2.0)
    return float(np.real(np.vdot(v, rho2 @ v)))


def chsh_S_max_from_rho2(rho2: np.ndarray) -> float:
    """
    Compute maximal CHSH S value for a two-qubit state rho2
    using the Horodecki formula:

      S_max = 2 * sqrt(u1 + u2),

    where u1,u2 are the two largest eigenvalues of T^T T and
    T_ij = Tr[rho2 (sigma_i ⊗ sigma_j)].
    """
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    paulis = [sx, sy, sz]

    T = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            op = np.kron(paulis[i], paulis[j])
            val = np.trace(rho2 @ op)
            T[i, j] = float(np.real(val))

    U = T.T @ T
    evals = np.linalg.eigvalsh(U)
    evals = np.sort(np.real(evals))
    u1 = evals[-1]
    u2 = evals[-2]
    S_max = 2.0 * np.sqrt(max(u1 + u2, 0.0))
    return float(S_max)


@functools.lru_cache(maxsize=None)
def get_spin_sector_projectors_3spin():
    """
    Construct projectors onto total spin sectors S=1/2 and S=3/2
    for three spin-1/2 particles.

    Returns:
        P_1_2, P_3_2 (each 8x8), and S_tot2 (8x8).
    """
    # Single-spin operators S = 1/2 * sigma
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    # Kronecker structure for 3 spins
    S1x = np.kron(np.kron(sx, I2), I2)
    S1y = np.kron(np.kron(sy, I2), I2)
    S1z = np.kron(np.kron(sz, I2), I2)

    S2x = np.kron(np.kron(I2, sx), I2)
    S2y = np.kron(np.kron(I2, sy), I2)
    S2z = np.kron(np.kron(I2, sz), I2)

    S3x = np.kron(np.kron(I2, I2), sx)
    S3y = np.kron(np.kron(I2, I2), sy)
    S3z = np.kron(np.kron(I2, I2), sz)

    Sx_tot = S1x + S2x + S3x
    Sy_tot = S1y + S2y + S3y
    Sz_tot = S1z + S2z + S3z

    S_tot2 = Sx_tot @ Sx_tot + Sy_tot @ Sy_tot + Sz_tot @ Sz_tot

    evals, evecs = np.linalg.eigh(S_tot2)
    P_1_2 = np.zeros_like(S_tot2)
    P_3_2 = np.zeros_like(S_tot2)
    for val, vec in zip(evals, evecs.T):
        if abs(val - 3.0 / 4.0) < 1e-6:      # S=1/2 -> S(S+1)=3/4
            P_1_2 += np.outer(vec, np.conjugate(vec))
        elif abs(val - 15.0 / 4.0) < 1e-6:   # S=3/2 -> S(S+1)=15/4
            P_3_2 += np.outer(vec, np.conjugate(vec))

    return P_1_2, P_3_2, S_tot2


def spin_sector_weights_3spin(rho_spin: np.ndarray) -> Dict[str, float]:
    """
    Given rho_spin (8x8), compute weights in S_tot=1/2 and S_tot=3/2 sectors.
    """
    P_1_2, P_3_2, _ = get_spin_sector_projectors_3spin()
    w_1_2 = float(np.real(np.trace(rho_spin @ P_1_2)))
    w_3_2 = float(np.real(np.trace(rho_spin @ P_3_2)))
    return {"S_1_2": w_1_2, "S_3_2": w_3_2}


# =============================================================================
# S3 antisymmetry metrics
# =============================================================================


def permute_state_3body(
    psi: np.ndarray, params: ThreeFermion3DParams, perm: Tuple[int, int, int]
) -> np.ndarray:
    """
    Apply permutation of particle labels to the 3-body state psi.

    perm is a tuple of length 3 with entries in {0,1,2} describing
    how (new) labels map to (old) labels:

        y[i] = x[perm[i]]

    where x = ( (r1,s1), (r2,s2), (r3,s3) ) is the new tuple at which we
    evaluate the permuted state, and y is the old tuple whose amplitude
    we use.

    Example:
      perm=(1,0,2) is transposition (12):
        (P_12 psi)(x1,x2,x3) = psi(x2,x1,x3).
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim_1 = Ns * 2
    dim = dim_1 * dim_1 * dim_1

    psi = psi.reshape((dim,))
    psi_p = np.zeros_like(psi)

    for idx_new in range(dim):
        r1, s1, r2, s2, r3, s3 = decode_basis_3body(idx_new, Ns)
        x = [(r1, s1), (r2, s2), (r3, s3)]
        y0 = x[perm[0]]
        y1 = x[perm[1]]
        y2 = x[perm[2]]
        idx_old = encode_basis_3body(
            y0[0], y0[1], y1[0], y1[1], y2[0], y2[1], Ns
        )
        psi_p[idx_new] = psi[idx_old]

    return psi_p


def antisymmetry_metrics_3body(
    psi: np.ndarray, params: ThreeFermion3DParams
) -> Dict[str, Any]:
    """
    Compute S3 antisymmetry weight and pairwise antisymmetry metrics.

    - antisym_S3_weight: weight in totally antisymmetric representation,
      via projector:

        A = (1/6)( I - P12 - P13 - P23 + P123 + P132 )

    - pair_antisym["12"],["13"],["23"]:
      w_A^(ij) = ||psi - P_ij psi||^2 / ( ||psi - P_ij psi||^2 + ||psi + P_ij psi||^2 )
    """
    # Define permutations as maps from new label to old label index (0-based)
    perms = {
        "id": (0, 1, 2),
        "12": (1, 0, 2),
        "13": (2, 1, 0),
        "23": (0, 2, 1),
        "123": (1, 2, 0),   # 1->2,2->3,3->1
        "132": (2, 0, 1),   # 1->3,3->2,2->1
    }

    psi_id = psi.reshape(-1)
    psi_12 = permute_state_3body(psi_id, params, perms["12"])
    psi_13 = permute_state_3body(psi_id, params, perms["13"])
    psi_23 = permute_state_3body(psi_id, params, perms["23"])
    psi_123 = permute_state_3body(psi_id, params, perms["123"])
    psi_132 = permute_state_3body(psi_id, params, perms["132"])

    # Antisymmetric projection A psi
    psi_A = (
        psi_id
        - psi_12
        - psi_13
        - psi_23
        + psi_123
        + psi_132
    ) / 6.0

    w_AS = float(np.real(np.vdot(psi_A, psi_A)))

    # Pairwise antisymmetry weights
    pair_antisym = {}
    for label, psi_p in [("12", psi_12), ("13", psi_13), ("23", psi_23)]:
        psi_minus = psi_id - psi_p
        psi_plus = psi_id + psi_p
        num = float(np.real(np.vdot(psi_minus, psi_minus)))
        den = num + float(np.real(np.vdot(psi_plus, psi_plus)))
        wA = num / den if den > 1e-16 else 0.0
        pair_antisym[label] = wA

    return {
        "antisym_S3_weight": w_AS,
        "pair_antisym": pair_antisym,
    }


# =============================================================================
# High-level 3-body classification
# =============================================================================


def classify_threebody_state(
    params: ThreeFermion3DParams,
    overlaps: Dict[str, float],
    rho_tot: np.ndarray,
    antisym: Dict[str, Any],
    spin_rhos: Dict[str, np.ndarray],
    spin_sector: Dict[str, float],
) -> Dict[str, Any]:
    """
    Build a qualitative classification of the 3-body state:

      - S3_antisym_fermionic
      - two_plus_one_spatial
      - exists_singlet_core_pair + which pair
      - pairwise_bell_violation_any
      - verdict (string)
    """
    # Spatial inhomogeneity: max density vs mean
    Ns = params.Lx * params.Ly * params.Lz
    mean_rho = float(np.mean(rho_tot))
    max_rho = float(np.max(rho_tot))
    inhomog_ratio = max_rho / (mean_rho + 1e-16)

    overlap_double = overlaps["overlap_prob_double"]
    overlap_triple = overlaps["overlap_prob_triple"]

    # Simple "two+one" spatial heuristic:
    # - need noticeable inhomogeneity
    # - non-trivial double overlap
    # - strongly suppressed triple overlap
    two_plus_one_spatial = (
        inhomog_ratio > 1.2 and overlap_double > 0.05 and overlap_triple < 0.02
    )

    # Singlet fractions & CHSH for each pair
    singlets = {}
    chsh_vals = {}
    for label, rho2 in spin_rhos.items():
        singlets[label] = singlet_fraction_from_rho2(rho2)
        chsh_vals[label] = chsh_S_max_from_rho2(rho2)

    max_singlet_pair = max(singlets, key=singlets.get)
    max_singlet_val = singlets[max_singlet_pair]
    exists_singlet_core_pair = max_singlet_val > 0.7  # heuristic threshold

    pairwise_bell_violation_any = any(abs(S) > 2.0 + 1e-6 for S in chsh_vals.values())

    # S3 antisymmetry
    w_AS = antisym["antisym_S3_weight"]
    S3_antisym_fermionic = w_AS > 0.9

    # Verdict logic
    if S3_antisym_fermionic and two_plus_one_spatial and exists_singlet_core_pair:
        verdict = "three_body_fermion_like_2plus1"
    elif two_plus_one_spatial and exists_singlet_core_pair:
        verdict = "three_body_entangled_2plus1_but_not_fully_antisymmetric"
    elif S3_antisym_fermionic and exists_singlet_core_pair:
        verdict = "three_body_fermion_like_spin_core_no_clear_spatial_2plus1"
    else:
        verdict = "three_body_mixed_or_nonfermionic"

    return {
        "inhomog_ratio": inhomog_ratio,
        "two_plus_one_spatial": two_plus_one_spatial,
        "exists_singlet_core_pair": exists_singlet_core_pair,
        "core_pair": max_singlet_pair,
        "pairwise_bell_violation_any": pairwise_bell_violation_any,
        "singlet_fractions": singlets,
        "pair_CHSH": chsh_vals,
        "spin_sector_weights": spin_sector,
        "S3_antisym_fermionic": S3_antisym_fermionic,
        "verdict": verdict,
    }


# =============================================================================
# Main experiment driver
# =============================================================================


def run_threefermion3d_experiment(
    params: ThreeFermion3DParams,
) -> Dict[str, Any]:
    print("======================================================================")
    print("3D THREE-EXCITATION SUBSTRATE (fermion-like model, 3-body)")
    print("======================================================================")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("----------------------------------------------------------------------")

    # Build Hamiltonian
    H = build_threefermion3d_hamiltonian(params)
    dim = H.shape[0]
    print(f"[INFO] Hilbert dimension (three excitations) = {dim}")
    print("[INFO] Solving for ground state (smallest eigenvalue) with eigsh...")
    evals, evecs = eigsh(
        H, k=params.k_eigs, which="SA", maxiter=params.max_eigsh_iter
    )
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    # Normalize
    norm = np.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    print(f"[RESULT] Ground state energy E0 = {E0:.6f}")
    print("----------------------------------------------------------------------")

    # Basic diagnostics
    E_gauss = gauss_energy_expectation_3body(psi0, params)
    overlaps = overlap_metrics_3body(psi0, params)
    rho1, rho2, rho3, rho_tot = site_densities_3body(psi0, params)

    print("Gauss-like energy (expectation):")
    print(f"  <H_gauss> = {E_gauss:.6f}")
    print()
    print("Overlap diagnostics (3-body):")
    print(
        f"  Double-overlap prob (>=2 on same site) = {overlaps['overlap_prob_double']:.6f}"
    )
    print(
        f"  Triple-overlap prob (r1==r2==r3)       = {overlaps['overlap_prob_triple']:.6f}"
    )
    print()

    print("Lump diagnostics (3D site densities, rho_tot):")
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    for r in range(Ns):
        x, y, z = site_coords_3d(r, Lx, Ly, Lz)
        print(
            f"  Site (x={x}, y={y}, z={z}): "
            f"rho1={rho1[r]:.4f}, rho2={rho2[r]:.4f}, rho3={rho3[r]:.4f}, rho_tot={rho_tot[r]:.4f}"
        )
    print()

    # Spin & antisymmetry diagnostics
    rho_spin = build_rho_spin_3body(psi0, params)
    spin_rhos = pair_spin_rdm_from_rho_spin(rho_spin)
    spin_sector = spin_sector_weights_3spin(rho_spin)
    antisym = antisymmetry_metrics_3body(psi0, params)
    classification = classify_threebody_state(
        params, overlaps, rho_tot, antisym, spin_rhos, spin_sector
    )

    print("Exchange antisymmetry diagnostics (S3):")
    print(f"  antisym_S3_weight = {antisym['antisym_S3_weight']:.6f}")
    print("  pair_antisym weights:")
    for lab, wA in antisym["pair_antisym"].items():
        print(f"    pair {lab}: w_A = {wA:.6f}")
    print()

    print("Spin diagnostics (pairwise):")
    print("  Pair singlet fractions F_S^(ij):")
    for lab, Fs in classification["singlet_fractions"].items():
        print(f"    pair {lab}: F_S = {Fs:.6f}")
    print("  Pair CHSH S_ij (max):")
    for lab, S_val in classification["pair_CHSH"].items():
        print(f"    pair {lab}: S = {S_val:.6f}")
    print()

    print("Spin sector weights (S_tot):")
    print(
        f"  w(S=1/2) = {classification['spin_sector_weights']['S_1_2']:.6f}, "
        f"w(S=3/2) = {classification['spin_sector_weights']['S_3_2']:.6f}"
    )
    print()

    print("Three-body classification:")
    print(f"  S3_antisym_fermionic      = {classification['S3_antisym_fermionic']}")
    print(f"  two_plus_one_spatial      = {classification['two_plus_one_spatial']}")
    print(
        f"  exists_singlet_core_pair  = {classification['exists_singlet_core_pair']} "
        f"(core_pair={classification['core_pair']})"
    )
    print(
        f"  pairwise_bell_violation_any = "
        f"{classification['pairwise_bell_violation_any']}"
    )
    print(f"  inhomog_ratio (max/mean rho_tot) = {classification['inhomog_ratio']:.6f}")
    print(f"  Verdict: {classification['verdict']}")
    print("======================================================================")

    return {
        "E0": E0,
        "E_gauss": E_gauss,
        "overlaps": overlaps,
        "rho1": rho1,
        "rho2": rho2,
        "rho3": rho3,
        "rho_tot": rho_tot,
        "rho_spin": rho_spin,
        "spin_rhos": spin_rhos,
        "spin_sector": spin_sector,
        "antisymmetry": antisym,
        "classification": classification,
        "params": asdict(params),
    }


# =============================================================================
# Command-line interface
# =============================================================================


def main():
    p = argparse.ArgumentParser(
        description=(
            "3D three-excitation substrate engine (threefermion_engine_3d). "
            "Build H and compute the 3-body ground state and diagnostics."
        )
    )

    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--Lz", type=int, default=2)

    p.add_argument("--J-hop", type=float, default=1.0, dest="J_hop")
    p.add_argument("--mass", type=float, default=0.1, dest="m")

    p.add_argument("--g-defrag", type=float, default=1.0, dest="g_defrag")
    p.add_argument("--sigma-defrag", type=float, default=1.0, dest="sigma_defrag")

    p.add_argument("--lambda-G", type=float, default=5.0, dest="lambda_G")
    p.add_argument("--lambda-S", type=float, default=-1.0, dest="lambda_S")
    p.add_argument("--lambda-T", type=float, default=0.0, dest="lambda_T")
    p.add_argument("--J-exch", type=float, default=1.0, dest="J_exch")

    p.add_argument("--max-eigsh-iter", type=int, default=5000, dest="max_eigsh_iter")
    p.add_argument("--k-eigs", type=int, default=1, dest="k_eigs")

    args = p.parse_args()

    params = ThreeFermion3DParams(
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
        max_eigsh_iter=args.max_eigsh_iter,
        k_eigs=args.k_eigs,
    )

    run_threefermion3d_experiment(params)


if __name__ == "__main__":
    main()
