#!/usr/bin/env python3
"""
ur_substrate_quantum_lab_v2.py

A unified "ur-script" lab tying together:

1) Finite-Hilbert "substrate" with two excitations
   ------------------------------------------------
   - 2D periodic lattice (minimal: 2x2) with Ns = Lx * Ly sites.
   - Two distinguishable excitations with spin-1/2 (↑, ↓).
   - Hilbert basis: |r1, s1; r2, s2>, r in {0..Ns-1}, s in {↑, ↓}.
   - Hamiltonian terms:
       * H_hop: tight-binding hopping on the lattice (J_hop).
       * H_mass: local mass term (m).
       * H_defrag: "defrag" potential encouraging clumping near lattice center.
       * H_gauss: Gauss-like penalty based on local occupancy fluctuations
                  (overlap penalty / information-flux constraint).
       * H_contact: spin contact at overlap (r1 == r2) that favors singlet
                    and penalizes triplet, echoing the 1D two-particle toy.

   - We compute the ground state and diagnose:
       * exchange antisymmetry score A under particle exchange,
       * probability of spatial overlap (r1 == r2),
       * singlet vs triplet composition at overlap,
       * contribution of the Gauss-like term.

   This is a minimal, fully quantum "substrate" sector where
   we can ask: does a constraint + defrag + contact spin structure
   push the ground state toward an antisymmetric (fermion-like) sector?

2) CHSH dimer (two-qubit quantum core)
   ------------------------------------
   - Two spins with a Heisenberg-like coupling.
   - Time evolution and CHSH S(t).
   - Demonstrates genuine Bell violations (S ≈ 2.828).

3) Fermion toy (exchange antisymmetry)
   -----------------------------------
   - Two-site Hilbert space with two excitations.
   - Symmetric vs antisymmetric states under swap.
   - Explicit eigenvalues +1 (boson-like) and -1 (fermion-like).

This is an ongoing playground, not a final theory.
"""

import argparse
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# QuTiP for CHSH + fermion toy
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
# SECTION 1: FINITE-HILBERT SUBSTRATE WITH TWO EXCITATIONS
# =============================================================================


@dataclass
class SubstrateParams:
    Lx: int = 2          # lattice size in x
    Ly: int = 2          # lattice size in y
    J_hop: float = 1.0   # hopping strength
    m: float = 0.1       # "mass" term per excitation
    g_defrag: float = 1.0  # defrag strength
    sigma_defrag: float = 1.0  # width for defrag potential
    lambda_G: float = 5.0   # Gauss-like penalty strength
    lambda_S: float = -1.0  # singlet bonus at overlap
    lambda_T: float = 0.0   # triplet penalty at overlap (parallel spins)
    J_exch: float = 1.0     # Heisenberg-like exchange at overlap
    max_eigsh_iter: int = 5000


# --- lattice helpers ---------------------------------------------------------


def site_index(x: int, y: int, Lx: int, Ly: int) -> int:
    """Map (x, y) -> r index."""
    return x + Lx * y


def site_coords(r: int, Lx: int, Ly: int) -> Tuple[int, int]:
    """Map r index -> (x, y)."""
    x = r % Lx
    y = r // Lx
    return x, y


def build_neighbors(Lx: int, Ly: int) -> Dict[int, List[int]]:
    """
    Build nearest-neighbor list for a 2D periodic lattice.
    """
    Ns = Lx * Ly
    neighbors = {r: [] for r in range(Ns)}
    for y in range(Ly):
        for x in range(Lx):
            r = site_index(x, y, Lx, Ly)
            # periodic neighbors
            xp = (x + 1) % Lx
            xm = (x - 1) % Lx
            yp = (y + 1) % Ly
            ym = (y - 1) % Ly
            neighbors[r].append(site_index(xp, y, Lx, Ly))
            neighbors[r].append(site_index(xm, y, Lx, Ly))
            neighbors[r].append(site_index(x, yp, Lx, Ly))
            neighbors[r].append(site_index(x, ym, Lx, Ly))
    return neighbors


def defrag_potential(r: int, params: SubstrateParams) -> float:
    """
    A simple "defrag" potential that favors sites near the center.

    We use a Gaussian in the discrete coordinates:

      V_defrag(r) = -exp(-dist2 / (2 sigma^2))

    So g_defrag * V_defrag favors clumping toward the center.
    """
    Lx, Ly = params.Lx, params.Ly
    x, y = site_coords(r, Lx, Ly)
    cx = 0.5 * (Lx - 1)
    cy = 0.5 * (Ly - 1)
    dx = x - cx
    dy = y - cy
    dist2 = dx * dx + dy * dy
    if params.sigma_defrag <= 0.0:
        return 0.0
    return -math.exp(-dist2 / (2.0 * params.sigma_defrag ** 2))


# --- basis indexing ----------------------------------------------------------


def encode_basis(r1: int, s1: int, r2: int, s2: int, Ns: int) -> int:
    """
    Encode basis index:

      r1 in [0, Ns-1], s1 in {0,1}, r2 in [0, Ns-1], s2 in {0,1}.

    We map (r1, s1, r2, s2) -> integer in [0, dim-1].

    dim = Ns * 2 * Ns * 2
    """
    return (((r1 * 2 + s1) * Ns * 2) + (r2 * 2 + s2))


def decode_basis(idx: int, Ns: int) -> Tuple[int, int, int, int]:
    """
    Inverse of encode_basis.
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


# --- Hamiltonian building ----------------------------------------------------


def build_substrate_hamiltonian(params: SubstrateParams) -> csr_matrix:
    """
    Build the two-excitation Hamiltonian on a 2D periodic lattice:

      H = H_hop + H_mass + H_defrag + H_gauss + H_contact

    Basis: |r1, s1; r2, s2>.
    """
    Lx, Ly = params.Lx, params.Ly
    Ns = Lx * Ly
    dim = Ns * 2 * Ns * 2

    neighbors = build_neighbors(Lx, Ly)
    H = lil_matrix((dim, dim), dtype=np.complex128)

    # Precompute defrag potentials
    V_defrag_site = np.array([defrag_potential(r, params) for r in range(Ns)], dtype=float)

    # Background occupancy for Gauss-like term: total 2 particles
    rho0 = 2.0 / Ns

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis(idx, Ns)

        # ---------------------------------------------------------------------
        # H_mass: each excitation gets +m
        # ---------------------------------------------------------------------
        H[idx, idx] += 2.0 * params.m

        # ---------------------------------------------------------------------
        # H_defrag: g_defrag * (V(r1) + V(r2))
        # ---------------------------------------------------------------------
        H[idx, idx] += params.g_defrag * (V_defrag_site[r1] + V_defrag_site[r2])

        # ---------------------------------------------------------------------
        # H_gauss: λ_G * 0.5 * sum_r (occ(r) - rho0)^2
        # occ(r) in {0,1,2}
        # ---------------------------------------------------------------------
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        G = occ.astype(float) - rho0
        gauss_energy = 0.5 * params.lambda_G * np.sum(G * G)
        H[idx, idx] += gauss_energy

        # ---------------------------------------------------------------------
        # H_contact: active when r1 == r2
        # - Heisenberg-like exchange J_exch S1·S2
        # - singlet bonus λ_S
        # - triplet penalty λ_T (parallel spins)
        #
        # We work in spin basis (s1,s2) ∈ {↑,↓}^2 = {(0,0),(0,1),(1,0),(1,1)}
        # with s^z = ±1/2 and S1·S2 having eigenvalues:
        #   -3/4 for singlet
        #   +1/4 for triplet
        #
        # We implement the contact term analogously to the 1D toy:
        #   - diag from Sz Sz and parallel-spin penalties,
        #   - off-diag flips for (↑↓ ↔ ↓↑) to favor singlet.
        # ---------------------------------------------------------------------
        if r1 == r2:
            # s_z mapping: 0 -> +1/2 (↑), 1 -> -1/2 (↓)
            sz1 = +0.5 if s1 == 0 else -0.5
            sz2 = +0.5 if s2 == 0 else -0.5

            # Heisenberg Sz1 Sz2 piece
            H[idx, idx] += params.J_exch * (sz1 * sz2)

            # Triplet penalty for parallel spins
            if s1 == s2:
                H[idx, idx] += params.lambda_T

            # Singlet bonus & spin-flip exchange for opposite spins
            if s1 != s2:
                # diagonal singlet bonus
                H[idx, idx] += params.lambda_S

                # off-diagonal flip: (↑↓) ↔ (↓↑)
                s1p, s2p = s2, s1  # flip spins
                idx_flip = encode_basis(r1, s1p, r2, s2p, Ns)
                # Exchange term (S+S- + S-S+), put a factor J_exch/2 for off-diagonal
                H[idx_flip, idx] += 0.5 * params.J_exch

        # ---------------------------------------------------------------------
        # H_hop: each particle hops to nearest neighbors
        # ---------------------------------------------------------------------
        # Hopping for particle 1
        for r1p in neighbors[r1]:
            idx_new = encode_basis(r1p, s1, r2, s2, Ns)
            H[idx_new, idx] += -params.J_hop

        # Hopping for particle 2
        for r2p in neighbors[r2]:
            idx_new = encode_basis(r1, s1, r2p, s2, Ns)
            H[idx_new, idx] += -params.J_hop

    return H.tocsr()


# --- diagnostics -------------------------------------------------------------


def antisymmetry_metrics(psi: np.ndarray, params: SubstrateParams) -> Dict[str, float]:
    """
    Compute antisymmetry metrics for the ground state wavefunction psi.

    psi is a flattened state of length dim = Ns*2*Ns*2.

    We compute:
      - total_norm
      - antisym_violation = ||psi + P_ex psi||^2
      - sym_violation = ||psi - P_ex psi||^2
      - antisym_score A = 1 - antisym_violation / total_norm
    """
    Lx, Ly = params.Lx, params.Ly
    Ns = Lx * Ly
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    total_norm = float(np.vdot(psi, psi).real)

    antisym_violation = 0.0
    sym_violation = 0.0

    for idx in range(dim):
        # decode
        r1, s1, r2, s2 = decode_basis(idx, Ns)
        idx_ex = encode_basis(r2, s2, r1, s1, Ns)

        psi_ij = psi[idx]
        psi_ji = psi[idx_ex]

        antisym_violation += abs(psi_ij + psi_ji) ** 2
        sym_violation += abs(psi_ij - psi_ji) ** 2

    antisym_score = 1.0 - antisym_violation / total_norm
    sym_score = 1.0 - sym_violation / total_norm

    return {
        "total_norm": total_norm,
        "antisym_violation": float(antisym_violation),
        "sym_violation": float(sym_violation),
        "antisym_score": float(antisym_score),
        "sym_score": float(sym_score),
    }


def overlap_and_spin_metrics(psi: np.ndarray, params: SubstrateParams) -> Dict[str, float]:
    """
    Analyze spatial overlap (r1 == r2) and spin composition at overlap.

    We compute:
      - overlap_prob: prob that r1 == r2.
      - singlet_same_site: total singlet norm at r1 == r2.
      - triplet_same_site: total triplet norm at r1 == r2.
      - singlet_fraction = singlet_same_site / (singlet_same_site + triplet_same_site)
        when the denominator is nonzero.
    """
    Lx, Ly = params.Lx, params.Ly
    Ns = Lx * Ly
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))

    overlap_prob = 0.0
    singlet_same_site = 0.0
    triplet_same_site = 0.0

    for r in range(Ns):
        # components at same site r1 = r2 = r
        # spin basis ordering: (s1, s2) ∈ { (0,0), (0,1), (1,0), (1,1) }
        amp_uu = psi[encode_basis(r, 0, r, 0, Ns)]  # |↑↑>
        amp_ud = psi[encode_basis(r, 0, r, 1, Ns)]  # |↑↓>
        amp_du = psi[encode_basis(r, 1, r, 0, Ns)]  # |↓↑>
        amp_dd = psi[encode_basis(r, 1, r, 1, Ns)]  # |↓↓>

        # Probability weight at this site
        site_prob = (
            abs(amp_uu) ** 2
            + abs(amp_ud) ** 2
            + abs(amp_du) ** 2
            + abs(amp_dd) ** 2
        )
        overlap_prob += site_prob

        # Singlet: (|↑↓> - |↓↑>)/sqrt(2)
        sing_amp = (amp_ud - amp_du) / math.sqrt(2.0)

        # Triplet m=+1: |↑↑>
        trip_m1 = amp_uu
        # Triplet m=0: (|↑↓> + |↓↑>)/sqrt(2)
        trip_m0 = (amp_ud + amp_du) / math.sqrt(2.0)
        # Triplet m=-1: |↓↓>
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


def gauss_energy_expectation(psi: np.ndarray, params: SubstrateParams) -> float:
    """
    Compute the expectation value of the Gauss-like penalty term:

      H_gauss = λ_G * 0.5 * sum_r (occ(r) - rho0)^2

    for the given state psi.
    """
    Lx, Ly = params.Lx, params.Ly
    Ns = Lx * Ly
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    rho0 = 2.0 / Ns

    E_gauss = 0.0
    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) < 1e-14:
            continue

        r1, s1, r2, s2 = decode_basis(idx, Ns)
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        G = occ.astype(float) - rho0
        gauss_energy = 0.5 * params.lambda_G * np.sum(G * G)
        E_gauss += gauss_energy * abs(amp) ** 2

    return float(E_gauss)


def run_substrate_ground_state(params: SubstrateParams) -> Dict[str, float]:
    """
    Build the substrate Hamiltonian and compute the ground state
    in the two-excitation sector, along with diagnostics.
    """
    print("=" * 80)
    print("FINITE-HILBERT SUBSTRATE: TWO-EXCITATION SECTOR")
    print("=" * 80)
    print("Substrate parameters:")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("-" * 80)

    H = build_substrate_hamiltonian(params)
    dim = H.shape[0]
    print(f"[INFO] Hilbert dimension (two excitations) = {dim}")
    print("[INFO] Solving for ground state (smallest eigenvalue)...")

    # Use eigsh for sparse Hermitian
    evals, evecs = eigsh(H, k=1, which="SA", maxiter=params.max_eigsh_iter)
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    print(f"[RESULT] Ground state energy E0 = {E0:.6f}")

    # Normalize psi0 just to be safe
    norm = math.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    anti = antisymmetry_metrics(psi0, params)
    overlap = overlap_and_spin_metrics(psi0, params)
    E_gauss = gauss_energy_expectation(psi0, params)

    print("-" * 80)
    print("Exchange antisymmetry diagnostics:")
    print(f"  Antisymmetry score A = {anti['antisym_score']:.6f}")
    print(f"  Symmetry score S     = {anti['sym_score']:.6f}")
    print(f"  Antisym violation    = {anti['antisym_violation']:.6e}")
    print()
    print("Overlap & spin diagnostics:")
    print(f"  Spatial overlap prob (r1 == r2)     = {overlap['overlap_prob']:.6f}")
    print(f"  Singlet weight at overlap           = {overlap['singlet_same_site']:.6f}")
    print(f"  Triplet weight at overlap           = {overlap['triplet_same_site']:.6f}")
    print(f"  Singlet fraction (overlap region)   = {overlap['singlet_fraction']:.6f}")
    print()
    print("Gauss-like energy (expectation):")
    print(f"  <H_gauss> = {E_gauss:.6f}")
    print("-" * 80)

    # Simple verdict
    if anti["antisym_score"] > 0.95:
        print("[VERDICT] Ground state is strongly exchange-antisymmetric (fermion-like) "
              "within this toy model.")
    elif overlap["overlap_prob"] < 0.05 and overlap["singlet_fraction"] > 0.9:
        print("[VERDICT] Ground state strongly suppresses overlap and prefers singlets "
              "when overlap occurs (partial fermion-like behavior).")
    else:
        print("[VERDICT] No strong emergent antisymmetry in this parameter regime; "
              "behavior is mixed.")
    print("=" * 80)

    return {
        "E0": E0,
        "antisym_score": anti["antisym_score"],
        "sym_score": anti["sym_score"],
        "overlap_prob": overlap["overlap_prob"],
        "singlet_fraction": overlap["singlet_fraction"],
        "E_gauss": E_gauss,
    }


# =============================================================================
# SECTION 2: CHSH DIMER (BELL VIOLATIONS)
# =============================================================================


@dataclass
class CHSHParams:
    J: float = 1.0
    hz: float = 0.0
    t_max: float = 5.0
    n_steps: int = 50
    initial_state: str = "bell"  # "bell" or "product"


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

    for psi_t in result.states:
        E_AB = float(expect(A1 * B2, psi_t))
        E_ABp = float(expect(A1 * B2p, psi_t))
        E_ApB = float(expect(A1p * B2, psi_t))
        E_ApBp = float(expect(A1p * B2p, psi_t))
        S = E_AB + E_ABp + E_ApB - E_ApBp
        S_vals.append(S)

    return {
        "times": list(times),
        "S": S_vals,
    }


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

    # Match tensor dims to a two-qubit operator
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
            "Ur-substrate quantum lab v2: finite-Hilbert substrate "
            "+ CHSH dimer + fermion toy."
        )
    )

    # Which experiments to run
    parser.add_argument(
        "--run-substrate",
        action="store_true",
        help="Run the finite-Hilbert substrate two-excitation test.",
    )
    parser.add_argument(
        "--no-run-substrate",
        action="store_true",
        help="Disable substrate test (overrides --run-substrate).",
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
    parser.add_argument("--Lx", type=int, default=2, help="Lattice size in x.")
    parser.add_argument("--Ly", type=int, default=2, help="Lattice size in y.")
    parser.add_argument("--J-hop", type=float, default=1.0, dest="J_hop",
                        help="Hopping strength J_hop.")
    parser.add_argument("--m", type=float, default=0.1, help="Mass term per excitation.")
    parser.add_argument("--g-defrag", type=float, default=1.0,
                        help="Defrag strength g_defrag.")
    parser.add_argument("--sigma-defrag", type=float, default=1.0,
                        help="Defrag Gaussian width.")
    parser.add_argument("--lambda-G", type=float, default=5.0, dest="lambda_G",
                        help="Gauss-like penalty strength.")
    parser.add_argument("--lambda-S", type=float, default=-1.0, dest="lambda_S",
                        help="Singlet bonus at overlap.")
    parser.add_argument("--lambda-T", type=float, default=0.0, dest="lambda_T",
                        help="Triplet penalty at overlap (parallel spins).")
    parser.add_argument("--J-exch", type=float, default=1.0, dest="J_exch",
                        help="Heisenberg-like exchange strength at overlap.")

    # CHSH parameters
    parser.add_argument("--chsh-J", type=float, default=1.0, help="CHSH exchange J.")
    parser.add_argument("--chsh-hz", type=float, default=0.0, help="CHSH z-field hz.")
    parser.add_argument("--chsh-t-max", type=float, default=5.0,
                        help="CHSH max evolution time.")
    parser.add_argument("--chsh-n-steps", type=int, default=50,
                        help="CHSH number of time steps.")
    parser.add_argument(
        "--chsh-initial-state",
        type=str,
        default="bell",
        choices=["bell", "product"],
        help="CHSH initial state.",
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

    # --- Substrate ---
    if run_substrate:
        sub_params = SubstrateParams(
            Lx=args.Lx,
            Ly=args.Ly,
            J_hop=args.J_hop,
            m=args.m,
            g_defrag=args.g_defrag,
            sigma_defrag=args.sigma_defrag,
            lambda_G=args.lambda_G,
            lambda_S=args.lambda_S,
            lambda_T=args.lambda_T,
            J_exch=args.J_exch,
        )
        run_substrate_ground_state(sub_params)

    # --- CHSH ---
    if run_chsh:
        chsh_params = CHSHParams(
            J=args.chsh_J,
            hz=args.chsh_hz,
            t_max=args.chsh_t_max,
            n_steps=args.chsh_n_steps,
            initial_state=args.chsh_initial_state,
        )
        chsh_data = run_chsh_evolution(chsh_params)
        print_chsh_summary(chsh_params, chsh_data)

    # --- Fermion toy ---
    if run_fermion:
        run_fermion_toy()


if __name__ == "__main__":
    main()
