#!/usr/bin/env python3
"""
substrate_gauge3d_meanfield.py

3D finite-Hilbert substrate with TWO spin-1/2 excitations on a 3D lattice,
coupled to a U(1) gauge field treated in a mean-field / variational way.

Degrees of freedom (conceptual):
  - Matter sector: two excitations on a 3D lattice with spin-1/2.
  - Gauge sector: U(1) link variables on each nearest-neighbor bond.

Numerical approximation (Option 3):
  - Matter is treated exactly via sparse exact diagonalization.
  - Gauge field is approximated by classical link phases {theta_ell} obtained
    by a self-consistent minimization of the total energy:

        E_total(theta) = <psi(theta)| H_matter[theta] |psi(theta)> + H_gauge[theta]

    where
        H_matter[theta] includes Peierls phases e^{i q theta_ij} on hops,
        H_gauge[theta] = (kappa/2) sum_plaq [1 - cos(Φ_p)],
        Φ_p = oriented sum of link phases around a plaquette.

  - Self-consistency loop:
        1. Start with some initial theta_ell (e.g. 0).
        2. Build H_matter[theta], solve for ground state psi(theta).
        3. Compute gradient dE_total/dtheta_ell from matter link currents
           and plaquette fluxes.
        4. Update theta_ell <- theta_ell - eta * grad_ell.
        5. Repeat until convergence or max iterations.

This script focuses on the TWO-excitation sector (Gate 1).
It keeps your existing substrate structure (defrag, Gauss, spin contact)
and adds an active U(1) gauge background that can self-organize in response
to the matter state.

Usage example (Windows, 2x2x2 lattice):

  python substrate_gauge3d_meanfield.py --Lx 2 --Ly 2 --Lz 2 ^
      --J-hop 1.0 --mass 0.1 ^
      --g-defrag 1.0 --sigma-defrag 1.0 ^
      --lambda-G 5.0 --lambda-S -1.0 --lambda-T 0.0 --J-exch 1.0 ^
      --gauge-kappa 1.0 --gauge-charge 1.0 --gauge-eta 0.1 ^
      --gauge-iters 20 --max-eigsh-iter 5000 --k-eigs 1
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Any

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


# =============================================================================
# Parameter dataclass
# =============================================================================


@dataclass
class TwoFermion3DGaugeParams:
    # Lattice dimensions
    Lx: int = 2
    Ly: int = 2
    Lz: int = 2

    # Matter parameters
    J_hop: float = 1.0
    m: float = 0.1

    g_defrag: float = 1.0
    sigma_defrag: float = 1.0

    lambda_G: float = 5.0
    lambda_S: float = -1.0
    lambda_T: float = 0.0
    J_exch: float = 1.0

    # Gauge parameters
    gauge_kappa: float = 1.0      # flux stiffness
    gauge_charge: float = 1.0     # effective charge q
    gauge_eta: float = 0.1        # gradient descent step size
    gauge_iters: int = 20         # number of self-consistency iterations

    # ED parameters
    max_eigsh_iter: int = 5000
    k_eigs: int = 1


# =============================================================================
# Lattice geometry and gauge structures
# =============================================================================


@dataclass
class LatticeGeometry:
    Lx: int
    Ly: int
    Lz: int
    Ns: int
    neighbors: Dict[int, List[int]]
    link_map: Dict[Tuple[int, int], Tuple[int, int]]  # (r, rp) -> (link_idx, sign)
    n_links: int
    plaquettes: List[List[Tuple[int, int]]]  # list of [(link_idx, sign), ...]


def site_index_3d(x: int, y: int, z: int, Lx: int, Ly: int, Lz: int) -> int:
    return x + Lx * (y + Ly * z)


def site_coords_3d(r: int, Lx: int, Ly: int, Lz: int) -> Tuple[int, int, int]:
    x = r % Lx
    tmp = r // Lx
    y = tmp % Ly
    z = tmp // Ly
    return x, y, z


def build_lattice_geometry(Lx: int, Ly: int, Lz: int) -> LatticeGeometry:
    """
    Build neighbors, link_map, and plaquettes for a 3D lattice with periodic BC.

    - neighbors[r]: list of nearest neighbors of site r.
    - link_map[(r, rp)] = (ell, sign) where ell is link index and sign = +1 if
      (r->rp) is along the stored orientation, -1 if opposite.
    - plaquettes: minimal squares in xy, yz, zx planes with oriented link signs.
    """
    Ns = Lx * Ly * Lz

    neighbors: Dict[int, List[int]] = {r: [] for r in range(Ns)}
    link_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
    links: List[Tuple[int, int]] = []

    # Build links in +x, +y, +z directions
    def add_link(i: int, j: int):
        idx = len(links)
        links.append((i, j))
        # orientation i -> j is +1
        link_map[(i, j)] = (idx, +1)
        link_map[(j, i)] = (idx, -1)
        neighbors[i].append(j)
        neighbors[j].append(i)

    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                r = site_index_3d(x, y, z, Lx, Ly, Lz)
                # +x direction
                xp = (x + 1) % Lx
                rp = site_index_3d(xp, y, z, Lx, Ly, Lz)
                if (r, rp) not in link_map and (rp, r) not in link_map:
                    add_link(r, rp)
                # +y direction
                yp = (y + 1) % Ly
                rp = site_index_3d(x, yp, z, Lx, Ly, Lz)
                if (r, rp) not in link_map and (rp, r) not in link_map:
                    add_link(r, rp)
                # +z direction
                zp = (z + 1) % Lz
                rp = site_index_3d(x, y, zp, Lx, Ly, Lz)
                if (r, rp) not in link_map and (rp, r) not in link_map:
                    add_link(r, rp)

    n_links = len(links)

    # Build plaquettes in xy, yz, zx planes
    plaquettes: List[List[Tuple[int, int]]] = []

    # XY plaquettes: (x,y,z) -> (x+1,y,z) -> (x+1,y+1,z) -> (x,y+1,z) -> back
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                i = site_index_3d(x, y, z, Lx, Ly, Lz)
                j = site_index_3d((x + 1) % Lx, y, z, Lx, Ly, Lz)
                k = site_index_3d((x + 1) % Lx, (y + 1) % Ly, z, Lx, Ly, Lz)
                l = site_index_3d(x, (y + 1) % Ly, z, Lx, Ly, Lz)
                p_links: List[Tuple[int, int]] = []
                for (a, b) in [(i, j), (j, k), (k, l), (l, i)]:
                    ell, sgn = link_map[(a, b)]
                    p_links.append((ell, sgn))
                plaquettes.append(p_links)

    # YZ plaquettes: (x,y,z) -> (x,y+1,z) -> (x,y+1,z+1) -> (x,y,z+1)
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                i = site_index_3d(x, y, z, Lx, Ly, Lz)
                j = site_index_3d(x, (y + 1) % Ly, z, Lx, Ly, Lz)
                k = site_index_3d(x, (y + 1) % Ly, (z + 1) % Lz, Lx, Ly, Lz)
                l = site_index_3d(x, y, (z + 1) % Lz, Lx, Ly, Lz)
                p_links = []
                for (a, b) in [(i, j), (j, k), (k, l), (l, i)]:
                    ell, sgn = link_map[(a, b)]
                    p_links.append((ell, sgn))
                plaquettes.append(p_links)

    # ZX plaquettes: (x,y,z) -> (x,y,z+1) -> (x+1,y,z+1) -> (x+1,y,z)
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                i = site_index_3d(x, y, z, Lx, Ly, Lz)
                j = site_index_3d(x, y, (z + 1) % Lz, Lx, Ly, Lz)
                k = site_index_3d((x + 1) % Lx, y, (z + 1) % Lz, Lx, Ly, Lz)
                l = site_index_3d((x + 1) % Lx, y, z, Lx, Ly, Lz)
                p_links = []
                for (a, b) in [(i, j), (j, k), (k, l), (l, i)]:
                    ell, sgn = link_map[(a, b)]
                    p_links.append((ell, sgn))
                plaquettes.append(p_links)

    return LatticeGeometry(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        Ns=Ns,
        neighbors=neighbors,
        link_map=link_map,
        n_links=n_links,
        plaquettes=plaquettes,
    )


# =============================================================================
# Basis mapping: |r1,s1; r2,s2> <-> index
# =============================================================================


def encode_basis_2body(r1: int, s1: int, r2: int, s2: int, Ns: int) -> int:
    """
    2-body basis encoding.

    s1,s2 in {0,1} for (↑,↓).
    index ranges 0..dim-1, where dim = (Ns*2)^2.
    """
    dim_1 = Ns * 2
    return (r1 * 2 + s1) * dim_1 + (r2 * 2 + s2)


def decode_basis_2body(idx: int, Ns: int) -> Tuple[int, int, int, int]:
    """
    Inverse mapping of encode_basis_2body.
    """
    dim_1 = Ns * 2
    r1s1 = idx // dim_1
    r2s2 = idx % dim_1

    r1 = r1s1 // 2
    s1 = r1s1 % 2
    r2 = r2s2 // 2
    s2 = r2s2 % 2
    return r1, s1, r2, s2


# =============================================================================
# Defrag potential
# =============================================================================


def defrag_potential_3d(params: TwoFermion3DGaugeParams) -> np.ndarray:
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
# Hamiltonian construction with gauge phases
# =============================================================================


def build_twofermion3d_gauge_hamiltonian(
    params: TwoFermion3DGaugeParams,
    geom: LatticeGeometry,
    theta: np.ndarray,
) -> csr_matrix:
    """
    Build the 3D two-excitation Hamiltonian for a fixed gauge configuration theta:

        H_matter[theta] = H_hop[theta] + H_mass + H_defrag + H_Gauss + H_contact

    where:
      - H_hop[theta] has Peierls phases e^{i q theta_ell} on each link ell.
      - H_mass, H_defrag, H_Gauss, H_contact are as in the original 2-body engine.
    """
    Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
    Ns = geom.Ns
    neighbors = geom.neighbors
    link_map = geom.link_map

    dim_1 = Ns * 2
    dim = dim_1 * dim_1

    V_defrag = defrag_potential_3d(params)
    rho0 = 2.0 / Ns

    H = lil_matrix((dim, dim), dtype=np.complex128)

    J_hop = params.J_hop
    q = params.gauge_charge

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_2body(idx, Ns)

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

        # 4) Contact spin term when sites coincide
        if r1 == r2:
            # S^z = +1/2 for s=0 (↑), -1/2 for s=1 (↓)
            sz1 = +0.5 if s1 == 0 else -0.5
            sz2 = +0.5 if s2 == 0 else -0.5

            # Heisenberg diagonal piece: Sz1 * Sz2
            H[idx, idx] += params.J_exch * (sz1 * sz2)

            # Triplet penalty for equal spins
            if s1 == s2:
                H[idx, idx] += params.lambda_T

            # Singlet bonus + spin flip if opposite spins
            if s1 != s2:
                H[idx, idx] += params.lambda_S

                # flip spins: |↑↓> <-> |↓↑>
                s1p, s2p = s2, s1
                idx_flip = encode_basis_2body(r1, s1p, r2, s2p, Ns)
                H[idx_flip, idx] += 0.5 * params.J_exch

        # 5) Hopping for particle 1 with gauge phase
        for rp in neighbors[r1]:
            ell, sgn = link_map[(r1, rp)]
            phase = np.exp(1j * q * sgn * theta[ell])
            idx_new = encode_basis_2body(rp, s1, r2, s2, Ns)
            H[idx_new, idx] += -J_hop * phase

        # 6) Hopping for particle 2 with gauge phase
        for rp in neighbors[r2]:
            ell, sgn = link_map[(r2, rp)]
            phase = np.exp(1j * q * sgn * theta[ell])
            idx_new = encode_basis_2body(r1, s1, rp, s2, Ns)
            H[idx_new, idx] += -J_hop * phase

    return H.tocsr()


# =============================================================================
# Diagnostics: Gauss energy, antisymmetry, overlaps, spin, CHSH
# =============================================================================


def gauss_energy_expectation_2body(
    psi: np.ndarray, params: TwoFermion3DGaugeParams, geom: LatticeGeometry
) -> float:
    """
    Expectation value of Gauss-like penalty term for 2 excitations.
    """
    Ns = geom.Ns
    dim_1 = Ns * 2
    dim = dim_1 * dim_1

    psi = psi.reshape((dim,))
    rho0 = 2.0 / Ns

    E_gauss = 0.0
    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) < 1e-14:
            continue

        r1, s1, r2, s2 = decode_basis_2body(idx, Ns)
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        G = occ.astype(float) - rho0
        gauss_energy = 0.5 * params.lambda_G * np.sum(G * G)

        E_gauss += gauss_energy * (abs(amp) ** 2)

    return float(E_gauss)


def antisymmetry_scores_2body(
    psi: np.ndarray, geom: LatticeGeometry
) -> Dict[str, float]:
    """
    Compute exchange antisymmetry and symmetry scores for the 2-body state psi:

      A = ||psi - P12 psi||^2 / (||psi - P12 psi||^2 + ||psi + P12 psi||^2)
      S = ||psi + P12 psi||^2 / (||psi - P12 psi||^2 + ||psi + P12 psi||^2)

    so that A+S = 1 (up to rounding). Perfect antisymmetry => A=1,S=0.
    """
    Ns = geom.Ns
    dim_1 = Ns * 2
    dim = dim_1 * dim_1

    psi = psi.reshape((dim,))
    psi_swap = np.zeros_like(psi)

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_2body(idx, Ns)
        idx_swap = encode_basis_2body(r2, s2, r1, s1, Ns)
        psi_swap[idx] = psi[idx_swap]

    psi_minus = psi - psi_swap
    psi_plus = psi + psi_swap

    n_minus = float(np.real(np.vdot(psi_minus, psi_minus)))
    n_plus = float(np.real(np.vdot(psi_plus, psi_plus)))
    den = n_minus + n_plus
    if den < 1e-16:
        return {"A": 0.0, "S": 0.0, "violation": 0.0}

    A = n_minus / den
    S = n_plus / den
    return {"A": A, "S": S, "violation": float(abs(1.0 - (A + S)))}


def overlap_spin_diagnostics(
    psi: np.ndarray, geom: LatticeGeometry
) -> Dict[str, float]:
    """
    Overlap & spin diagnostics restricted to r1 == r2 region:

      - overlap_prob: P(r1 == r2)
      - singlet_weight: total probability weight in singlet subspace at overlap
      - triplet_weight: total probability weight in triplet subspace at overlap
      - singlet_fraction (overlap region) = singlet_weight / overlap_prob
    """
    Ns = geom.Ns
    dim_1 = Ns * 2
    dim = dim_1 * dim_1

    psi = psi.reshape((dim,))

    overlap_prob = 0.0
    singlet_weight = 0.0
    triplet_weight = 0.0

    # spin basis indices: 0=↑↑, 1=↑↓, 2=↓↑, 3=↓↓
    for r in range(Ns):
        amps = np.zeros(4, dtype=np.complex128)
        # ↑↑
        idx = encode_basis_2body(r, 0, r, 0, Ns)
        amps[0] = psi[idx]
        # ↑↓
        idx = encode_basis_2body(r, 0, r, 1, Ns)
        amps[1] = psi[idx]
        # ↓↑
        idx = encode_basis_2body(r, 1, r, 0, Ns)
        amps[2] = psi[idx]
        # ↓↓
        idx = encode_basis_2body(r, 1, r, 1, Ns)
        amps[3] = psi[idx]

        p_r = float(np.sum(np.abs(amps) ** 2))
        if p_r < 1e-16:
            continue

        overlap_prob += p_r

        # Singlet |S> = (|↑↓> - |↓↑>)/sqrt(2)
        S_amp = (amps[1] - amps[2]) / np.sqrt(2.0)
        singlet_weight += float(np.abs(S_amp) ** 2)

        # Triplet m=+1: |↑↑>
        T1_amp = amps[0]
        # Triplet m=0: (|↑↓> + |↓↑>)/sqrt(2)
        T0_amp = (amps[1] + amps[2]) / np.sqrt(2.0)
        # Triplet m=-1: |↓↓>
        Tm1_amp = amps[3]
        triplet_weight += float(
            np.abs(T1_amp) ** 2 + np.abs(T0_amp) ** 2 + np.abs(Tm1_amp) ** 2
        )

    singlet_fraction = singlet_weight / overlap_prob if overlap_prob > 1e-16 else 0.0

    return {
        "overlap_prob": overlap_prob,
        "singlet_weight": singlet_weight,
        "triplet_weight": triplet_weight,
        "singlet_fraction": singlet_fraction,
    }


def site_densities_2body(
    psi: np.ndarray, geom: LatticeGeometry
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one-body densities for each excitation and total:

      - rho1[r] = probability that excitation 1 is at site r
      - rho2[r] = probability that excitation 2 is at site r
      - rho_tot[r] = rho1[r] + rho2[r]
    """
    Ns = geom.Ns
    dim_1 = Ns * 2
    dim = dim_1 * dim_1

    psi = psi.reshape((dim,))

    rho1 = np.zeros(Ns, dtype=float)
    rho2 = np.zeros(Ns, dtype=float)

    for idx in range(dim):
        amp2 = abs(psi[idx]) ** 2
        if amp2 < 1e-16:
            continue

        r1, s1, r2, s2 = decode_basis_2body(idx, Ns)
        rho1[r1] += amp2
        rho2[r2] += amp2

    rho_tot = rho1 + rho2
    return rho1, rho2, rho_tot


def build_spin_rdm_2body(
    psi: np.ndarray, geom: LatticeGeometry
) -> np.ndarray:
    """
    Build the 2-spin reduced density matrix rho_spin (4x4) by tracing
    over positions:

      rho_spin[a,b] = sum_{r1,r2} psi(r1,s1; r2,s2) psi^*(r1,r2,s1',s2'),

    where a,b index spin basis |↑↑>,|↑↓>,|↓↑>,|↓↓>.
    """
    Ns = geom.Ns
    dim_1 = Ns * 2
    dim = dim_1 * dim_1

    psi = psi.reshape((dim,))

    rho_spin = np.zeros((4, 4), dtype=np.complex128)

    spin_blocks: Dict[Tuple[int, int], np.ndarray] = {}

    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) < 1e-16:
            continue

        r1, s1, r2, s2 = decode_basis_2body(idx, Ns)
        key = (r1, r2)
        vec = spin_blocks.get(key)
        if vec is None:
            vec = np.zeros(4, dtype=np.complex128)
            spin_blocks[key] = vec
        a = 2 * s1 + s2
        vec[a] += amp

    for vec in spin_blocks.values():
        rho_spin += np.outer(vec, np.conjugate(vec))

    tr = np.trace(rho_spin)
    if abs(tr) > 1e-14:
        rho_spin /= tr

    return rho_spin


def singlet_fraction_from_rho2(rho2: np.ndarray) -> float:
    """
    Compute singlet fraction F_S = <S|rho2|S> for two-spin RDM rho2,
    where |S> = (|↑↓> - |↓↑>)/sqrt(2).
    Basis for rho2 is |↑↑>,|↑↓>,|↓↑>,|↓↓>.
    """
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
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
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


# =============================================================================
# Gauge sector: energy and gradients
# =============================================================================


def gauge_energy(
    theta: np.ndarray, geom: LatticeGeometry, params: TwoFermion3DGaugeParams
) -> float:
    """
    Gauge energy H_gauge[theta] = (kappa/2) sum_plaq [1 - cos(Φ_p)],
    where Φ_p is the oriented sum of link phases around each plaquette.
    """
    kappa = params.gauge_kappa
    if kappa == 0.0:
        return 0.0

    E = 0.0
    for p in geom.plaquettes:
        phi = 0.0
        for ell, sgn in p:
            phi += sgn * theta[ell]
        E += (1.0 - np.cos(phi))
    return 0.5 * kappa * E


def gauge_energy_grad(
    theta: np.ndarray, geom: LatticeGeometry, params: TwoFermion3DGaugeParams
) -> np.ndarray:
    """
    Derivative of gauge energy with respect to theta_ell:

      dE_gauge/dtheta_ell = (kappa/2) * sum_p sin(Φ_p) * sign_p(ell),
    where the sum is over plaquettes containing link ell.
    """
    kappa = params.gauge_kappa
    n_links = geom.n_links
    grad = np.zeros(n_links, dtype=float)
    if kappa == 0.0:
        return grad

    for p in geom.plaquettes:
        phi = 0.0
        for ell, sgn in p:
            phi += sgn * theta[ell]
        s = np.sin(phi)
        if abs(s) < 1e-16:
            continue
        for ell, sgn in p:
            grad[ell] += sgn * s

    grad *= 0.5 * kappa
    return grad


def matter_link_currents(
    psi: np.ndarray,
    theta: np.ndarray,
    geom: LatticeGeometry,
    params: TwoFermion3DGaugeParams,
) -> np.ndarray:
    """
    Compute matter contribution to dE/dtheta_ell via link currents.

    For each directed hop r->rp across link ell with sign sgn:

      H_hop term ~ -J_hop e^{i q sgn theta_ell} |rp><r| + h.c.

    The derivative of <H_hop> w.r.t. theta_ell is:

      dE/dtheta_ell (matter) = q * J_ell,

    with

      J_ell = -2 J_hop sum_{hops along link ell} sgn * Im( e^{i q sgn theta_ell} * psi*_rp * psi_r )

    summed over both particles and all basis states.
    """
    Ns = geom.Ns
    neighbors = geom.neighbors
    link_map = geom.link_map
    dim_1 = Ns * 2
    dim = dim_1 * dim_1

    psi = psi.reshape((dim,))
    J_hop = params.J_hop
    q = params.gauge_charge

    n_links = geom.n_links
    J_links = np.zeros(n_links, dtype=float)

    # Loop over basis states and contribute to link currents from hopping terms
    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) < 1e-16:
            continue

        r1, s1, r2, s2 = decode_basis_2body(idx, Ns)

        # particle 1 hops
        for rp in neighbors[r1]:
            ell, sgn = link_map[(r1, rp)]
            idx_new = encode_basis_2body(rp, s1, r2, s2, Ns)
            amp_new = psi[idx_new]
            phase = np.exp(1j * q * sgn * theta[ell])
            contrib = -2.0 * J_hop * sgn * np.imag(np.conjugate(amp_new) * phase * amp)
            J_links[ell] += contrib

        # particle 2 hops
        for rp in neighbors[r2]:
            ell, sgn = link_map[(r2, rp)]
            idx_new = encode_basis_2body(r1, s1, rp, s2, Ns)
            amp_new = psi[idx_new]
            phase = np.exp(1j * q * sgn * theta[ell])
            contrib = -2.0 * J_hop * sgn * np.imag(np.conjugate(amp_new) * phase * amp)
            J_links[ell] += contrib

    # dE/dtheta_ell (matter) = q * J_links[ell]
    return q * J_links


# =============================================================================
# Main mean-field gauge + matter driver
# =============================================================================


def run_twofermion3d_gauge_meanfield(
    params: TwoFermion3DGaugeParams,
) -> Dict[str, Any]:
    print("======================================================================")
    print("3D TWO-EXCITATION SUBSTRATE WITH MEAN-FIELD U(1) GAUGE FIELD")
    print("======================================================================")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("----------------------------------------------------------------------")

    # Build lattice geometry once
    geom = build_lattice_geometry(params.Lx, params.Ly, params.Lz)
    Ns = geom.Ns
    dim_1 = Ns * 2
    dim = dim_1 * dim_1
    print(f"[INFO] Hilbert dimension (two excitations) = {dim}")
    print(f"[INFO] Number of gauge links = {geom.n_links}")
    print(f"[INFO] Number of plaquettes = {len(geom.plaquettes)}")
    print("----------------------------------------------------------------------")

    # Initialize gauge field phases
    theta = np.zeros(geom.n_links, dtype=float)

    best_result: Dict[str, Any] = {}
    best_E_total = None

    for it in range(params.gauge_iters):
        print(f"[GAUGE-ITER] {it+1}/{params.gauge_iters}")

        # 1) Build matter Hamiltonian for current theta
        H = build_twofermion3d_gauge_hamiltonian(params, geom, theta)

        # 2) Solve for ground state
        print("[INFO] Solving for ground state (smallest eigenvalue) with eigsh...")
        evals, evecs = eigsh(
            H, k=params.k_eigs, which="SA", maxiter=params.max_eigsh_iter
        )
        E0 = float(evals[0].real)
        psi0 = evecs[:, 0]

        # Normalize
        norm = np.sqrt(float(np.vdot(psi0, psi0).real))
        if norm > 0.0:
            psi0 /= norm

        # 3) Gauge energy and total energy
        Eg = gauge_energy(theta, geom, params)
        E_total = E0 + Eg

        # 4) Diagnostics on matter state
        E_gauss = gauss_energy_expectation_2body(psi0, params, geom)
        A_scores = antisymmetry_scores_2body(psi0, geom)
        overlap_info = overlap_spin_diagnostics(psi0, geom)
        rho1, rho2, rho_tot = site_densities_2body(psi0, geom)
        rho_spin = build_spin_rdm_2body(psi0, geom)
        Fs_spin = singlet_fraction_from_rho2(rho_spin)
        S_chsh = chsh_S_max_from_rho2(rho_spin)

        print(f"  [RESULT] Matter ground state energy E0   = {E0:.6f}")
        print(f"           Gauge energy E_gauge           = {Eg:.6f}")
        print(f"           Total energy E_total           = {E_total:.6f}")
        print(f"           Gauss-like energy <H_gauss>    = {E_gauss:.6f}")
        print("  Exchange antisymmetry diagnostics:")
        print(f"    Antisymmetry score A = {A_scores['A']:.6f}")
        print(f"    Symmetry score S     = {A_scores['S']:.6f}")
        print(f"    Antisym violation    = {A_scores['violation']:.6e}")
        print("  Overlap & spin diagnostics (r1 == r2):")
        print(
            f"    Spatial overlap prob (r1 == r2)     = {overlap_info['overlap_prob']:.6f}"
        )
        print(
            f"    Singlet weight at overlap           = {overlap_info['singlet_weight']:.6f}"
        )
        print(
            f"    Triplet weight at overlap           = {overlap_info['triplet_weight']:.6f}"
        )
        print(
            f"    Singlet fraction (overlap region)   = {overlap_info['singlet_fraction']:.6f}"
        )
        print("  Spin sector (2-qubit RDM):")
        print(f"    Singlet fraction F_S (global spin)  = {Fs_spin:.6f}")
        print("  CHSH diagnostics (spin sector):")
        print(f"    S = {S_chsh:.6f}, |S| = {abs(S_chsh):.6f}")
        print("    (|S| <= 2: local realistic, 2 < |S| <= 2√2: quantum-allowed)")

        fermionic_core = (
            A_scores["A"] > 0.95 and overlap_info["singlet_fraction"] > 0.95
        )
        bell_violation = abs(S_chsh) > 2.0 + 1e-6
        near_tsirelson = abs(abs(S_chsh) - 2.828427) < 0.05

        if fermionic_core:
            print(
                "  [VERDICT] Ground state has a strongly fermionic core "
                "(exchange-antisymmetric with singlet overlap)."
            )
        else:
            print(
                "  [VERDICT] Antisymmetry and/or singlet preference is partial "
                "or absent in this parameter regime."
            )

        if bell_violation:
            print(
                "  [VERDICT] Spin sector exhibits CHSH violation (Bell-inequality "
                "violation) in this reduced 2-qubit state."
            )
        else:
            print("  [VERDICT] No CHSH violation in the spin sector for these parameters.")

        if near_tsirelson:
            print(
                "  [VERDICT] Spin correlations are close to Tsirelson bound "
                "|S| ≈ 2.828 under the Horodecki optimization."
            )

        print("----------------------------------------------------------------------")

        # 5) Save best result so far
        if best_E_total is None or E_total < best_E_total:
            best_E_total = E_total
            best_result = {
                "E0": E0,
                "E_gauge": Eg,
                "E_total": E_total,
                "E_gauss": E_gauss,
                "antisymmetry": A_scores,
                "overlap": overlap_info,
                "rho1": rho1,
                "rho2": rho2,
                "rho_tot": rho_tot,
                "rho_spin": rho_spin,
                "Fs_spin": Fs_spin,
                "S_chsh": S_chsh,
                "fermionic_core": fermionic_core,
                "bell_violation": bell_violation,
                "near_tsirelson": near_tsirelson,
                "theta": theta.copy(),
            }

        # 6) Compute gradient and update gauge field
        grad_matter = matter_link_currents(psi0, theta, geom, params)
        grad_gauge = gauge_energy_grad(theta, geom, params)
        grad_total = grad_matter + grad_gauge

        max_grad = float(np.max(np.abs(grad_total))) if grad_total.size > 0 else 0.0
        print(f"  [GAUGE] max |dE/dtheta| = {max_grad:.6e}")

        # Update theta with gradient descent
        theta = theta - params.gauge_eta * grad_total

        # Wrap theta into (-pi, pi] for numerical stability
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

        print("----------------------------------------------------------------------")

    print("======================================================================")
    print("SELF-CONSISTENT GAUGE RUN COMPLETE")
    if best_E_total is not None:
        print(f"  Best E_total found = {best_E_total:.6f}")
    print("======================================================================")

    best_result["params"] = asdict(params)
    best_result["geom"] = {
        "Lx": geom.Lx,
        "Ly": geom.Ly,
        "Lz": geom.Lz,
        "Ns": geom.Ns,
        "n_links": geom.n_links,
        "n_plaquettes": len(geom.plaquettes),
    }

    return best_result


# =============================================================================
# Command-line interface
# =============================================================================


def main():
    p = argparse.ArgumentParser(
        description=(
            "3D two-excitation substrate engine with mean-field U(1) gauge "
            "field (substrate_gauge3d_meanfield)."
        )
    )

    # Lattice
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--Lz", type=int, default=2)

    # Matter parameters
    p.add_argument("--J-hop", type=float, default=1.0, dest="J_hop")
    p.add_argument("--mass", type=float, default=0.1, dest="m")

    p.add_argument("--g-defrag", type=float, default=1.0, dest="g_defrag")
    p.add_argument("--sigma-defrag", type=float, default=1.0, dest="sigma_defrag")

    p.add_argument("--lambda-G", type=float, default=5.0, dest="lambda_G")
    p.add_argument("--lambda-S", type=float, default=-1.0, dest="lambda_S")
    p.add_argument("--lambda-T", type=float, default=0.0, dest="lambda_T")
    p.add_argument("--J-exch", type=float, default=1.0, dest="J_exch")

    # Gauge parameters
    p.add_argument(
        "--gauge-kappa",
        type=float,
        default=1.0,
        dest="gauge_kappa",
        help="Gauge flux stiffness (kappa).",
    )
    p.add_argument(
        "--gauge-charge",
        type=float,
        default=1.0,
        dest="gauge_charge",
        help="Effective gauge charge q for excitations.",
    )
    p.add_argument(
        "--gauge-eta",
        type=float,
        default=0.1,
        dest="gauge_eta",
        help="Gradient descent step size for gauge field.",
    )
    p.add_argument(
        "--gauge-iters",
        type=int,
        default=20,
        dest="gauge_iters",
        help="Number of gauge-matter self-consistency iterations.",
    )

    # ED controls
    p.add_argument("--max-eigsh-iter", type=int, default=5000, dest="max_eigsh_iter")
    p.add_argument("--k-eigs", type=int, default=1, dest="k_eigs")

    args = p.parse_args()

    params = TwoFermion3DGaugeParams(
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
        gauge_kappa=args.gauge_kappa,
        gauge_charge=args.gauge_charge,
        gauge_eta=args.gauge_eta,
        gauge_iters=args.gauge_iters,
        max_eigsh_iter=args.max_eigsh_iter,
        k_eigs=args.k_eigs,
    )

    run_twofermion3d_gauge_meanfield(params)


if __name__ == "__main__":
    main()
