#!/usr/bin/env python3
"""
substrate_engine.py

Unified substrate lab / engine.

Includes:

1) DIRAC+EM GAUGE SYMMETRY LAB
   ----------------------------
   - Uses yee_dirac_substrate.py (2+1D Dirac–Maxwell on a Yee grid).
   - Evolves Psi and EM fields with minimal coupling.
   - Applies a local U(1) gauge transformation:
         Psi' = exp(i q chi) Psi
         Ax'  = Ax + ∂_x chi
         Ay'  = Ay + ∂_y chi
   - Compares covariant Dirac energy E_D before/after.

2) FINITE-HILBERT TWO-EXCITATION ANTISYMMETRY MODEL
   -------------------------------------------------
   - 2D periodic lattice with Ns = Lx * Ly sites, two spin-1/2 excitations.
   - Basis |r1, s1; r2, s2>, dim = Ns*2 * Ns*2.
   - Hamiltonian:
       H = H_hop + H_mass + H_defrag + H_Gauss + H_contact
     with:
       * hopping (J_hop),
       * mass m per excitation,
       * defrag potential (Gaussian center bias),
       * Gauss-like occupancy penalty,
       * local contact spin interaction at overlap.

   - Computes ground state, exchange antisymmetry, overlap probability,
     singlet/triplet composition at overlap, and Gauss-like energy.

3) CHSH DIMER / BELL-TEST CORE
   ---------------------------
   - Two-qubit Heisenberg dimer with tunable J, hz.
   - Initial Bell or product states.
   - Time evolution via QuTiP mesolve.
   - Computes CHSH S(t) and max S(t) over the time grid.

The module can be imported as a library OR run as a script:

    python substrate_engine.py
"""

import argparse
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np

# SciPy for finite-Hilbert diagonalization
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# QuTiP for CHSH dimer tests
from qutip import (
    basis,
    tensor,
    qeye,
    sigmax,
    sigmay,
    sigmaz,
    mesolve,
    expect,
)


# Import Dirac+EM substrate
import yee_dirac_substrate as yds


# =============================================================================
# SECTION 1: DIRAC+EM GAUGE SYMMETRY LAB
# =============================================================================

@dataclass
class DiracGaugeParams:
    Nx: int = 32
    Ny: int = 32
    dx: float = 1.0
    dt: float = 0.01
    steps: int = 200
    c: float = 1.0
    q: float = 1.0
    m: float = 0.1
    defrag: bool = False
    defrag_kappa: float = 0.1
    defrag_lambda: float = 1.0
    defrag_g: float = 0.5
    chi_amp: float = 0.5
    chi_mode_x: int = 1
    chi_mode_y: int = 1
    out_dir: str = "gauge_lab_output"


def build_chi_field(Ny, Nx, dx, amp, mode_x, mode_y, xp):
    """
    χ(x,y) = amp * sin(2π n_x x/Lx) * sin(2π n_y y/Ly)
    """
    Lx = Nx * dx
    Ly = Ny * dx

    x = xp.arange(Nx) * dx
    y = xp.arange(Ny) * dx
    X, Y = xp.meshgrid(x, y)

    kx = 2.0 * np.pi * mode_x / Lx
    ky = 2.0 * np.pi * mode_y / Ly

    chi = amp * xp.sin(kx * X) * xp.sin(ky * Y)
    return chi


def dirac_energy_covariant(Psi, dx, c, m, alpha1, alpha2, beta,
                           Ax, Ay, q, phi, g_phi, xp):
    """
    E_D = Re ∫ Ψ† H_D[A] Ψ d^2x
    where H_D is implemented by yee_dirac_substrate.dirac_hamiltonian_action.
    """
    Ny, Nx = Psi.shape[1], Psi.shape[2]
    dA = dx * dx

    H_Psi = yds.dirac_hamiltonian_action(
        Psi, dx, c, m, alpha1, alpha2, beta, xp,
        Ax=Ax, Ay=Ay, q=q,
        phi=phi, g_phi=g_phi
    )

    Psi_flat = Psi.reshape(2, Ny * Nx)
    H_Psi_flat = H_Psi.reshape(2, Ny * Nx)

    Psi_dag_flat = xp.conjugate(Psi_flat).T
    density_flat = xp.sum(Psi_dag_flat * H_Psi_flat.T, axis=1)
    E_complex = xp.sum(density_flat) * dA

    return float(E_complex.real)


def run_dirac_gauge_experiment(params: DiracGaugeParams) -> Dict[str, Any]:
    """
    Run the Dirac+EM gauge symmetry lab and return diagnostics in a dict.
    """
    xp = yds.xp
    XP_BACKEND = yds.XP_BACKEND

    Nx = params.Nx
    Ny = params.Ny
    dx = params.dx
    dt = params.dt
    steps = params.steps
    c = params.c
    q = params.q
    m = params.m

    defrag = params.defrag
    defrag_kappa = params.defrag_kappa
    defrag_lambda = params.defrag_lambda
    defrag_g = params.defrag_g if defrag else 0.0

    out_dir = params.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("======================================================================")
    print("DIRAC GAUGE SYMMETRY LAB (substrate_engine)")
    print("======================================================================")
    print(f"[INIT] Backend      : {XP_BACKEND}")
    print(f"[INIT] Grid         : Nx={Nx}, Ny={Ny}, dx={dx}, dt={dt}")
    print(f"[INIT] Steps (warm) : {steps}")
    print(f"[INIT] Parameters   : c={c}, q={q}, m={m}")
    if defrag:
        print(f"[INIT] Defrag ON    : kappa={defrag_kappa}, lambda={defrag_lambda}, g_phi={defrag_g}")
    else:
        print("[INIT] Defrag OFF")
    print(f"[INIT] Gauge phase  : chi_amp={params.chi_amp}, modes=({params.chi_mode_x},{params.chi_mode_y})")
    print("======================================================================")

    # Dirac matrices
    sigma_x, sigma_y, sigma_z, gamma0, gamma1, gamma2, alpha1, alpha2, beta = \
        yds.build_dirac_matrices(xp)

    # Initialize spinor & EM fields
    Psi_n = yds.init_spinor(Ny, Nx, dx, xp, m)
    Ex, Ey, Bz = yds.init_em_fields(Ny, Nx, xp)
    Ax = xp.zeros_like(Ex)
    Ay = xp.zeros_like(Ey)

    if defrag:
        phi = xp.zeros((Ny, Nx), dtype=xp.float64)
    else:
        phi = None

    # Euler step to get Psi^(1)
    H_Psi = yds.dirac_hamiltonian_action(
        Psi_n, dx, c, m, alpha1, alpha2, beta, xp,
        Ax=Ax, Ay=Ay, q=q,
        phi=phi, g_phi=defrag_g
    )
    Psi_np1 = Psi_n - 1j * dt * H_Psi
    norm = xp.sqrt(xp.sum(xp.abs(Psi_np1) ** 2))
    if norm != 0:
        Psi_np1 /= norm

    Psi_nm1 = Psi_n
    Psi_n = Psi_np1

    # Time evolution
    eps0 = 1.0
    print("[INFO] Warming up system with coupled Dirac+EM evolution...")
    t0 = time.time()
    for step in range(1, steps + 1):
        t = step * dt

        rho, Jx, Jy = yds.compute_spinor_current(Psi_n, alpha1, alpha2, q, c, xp)
        Ex, Ey, Bz = yds.update_em_fields(Ex, Ey, Bz, Jx, Jy, dx, dt, c, eps0, xp)

        Ax = Ax - dt * Ex
        Ay = Ay - dt * Ey

        if defrag:
            rho_sm = rho + defrag_lambda * yds.laplacian(rho, dx)
            phi = phi + dt * (-defrag_kappa * (phi - rho_sm))

        H_Psi_n = yds.dirac_hamiltonian_action(
            Psi_n, dx, c, m, alpha1, alpha2, beta, xp,
            Ax=Ax, Ay=Ay, q=q,
            phi=phi, g_phi=defrag_g
        )

        Psi_new = Psi_nm1 - 2j * dt * H_Psi_n
        norm = xp.sqrt(xp.sum(xp.abs(Psi_new) ** 2))
        if norm != 0:
            Psi_new /= norm

        Psi_nm1, Psi_n = Psi_n, Psi_new

        if step % max(1, steps // 5) == 0:
            print(f"[WARMUP] step={step:6d} t={t:7.3f}")

    t1 = time.time()
    print(f"[INFO] Warmup complete in {t1 - t0:.2f} s")

    # Energy before gauge transform
    print("[INFO] Computing Dirac covariant energy before gauge transform...")
    E_D_before = dirac_energy_covariant(
        Psi_n, dx, c, m, alpha1, alpha2, beta,
        Ax, Ay, q, phi, defrag_g, xp
    )
    norm_before = float(np.sqrt(float(yds.xp.sum(yds.xp.abs(Psi_n) ** 2))))

    # Build χ
    chi = build_chi_field(
        Ny, Nx, dx,
        amp=params.chi_amp,
        mode_x=params.chi_mode_x,
        mode_y=params.chi_mode_y,
        xp=xp
    )
    dchi_dx = yds.central_diff_x(chi, dx)
    dchi_dy = yds.central_diff_y(chi, dx)

    print("[INFO] Applying local U(1) gauge transformation...")
    phase = xp.exp(1j * q * chi)
    Psi_prime = Psi_n * phase
    Ax_prime = Ax + dchi_dx
    Ay_prime = Ay + dchi_dy

    print("[INFO] Computing Dirac covariant energy after gauge transform...")
    E_D_after = dirac_energy_covariant(
        Psi_prime, dx, c, m, alpha1, alpha2, beta,
        Ax_prime, Ay_prime, q, phi, defrag_g, xp
    )
    norm_after = float(np.sqrt(float(yds.xp.sum(yds.xp.abs(Psi_prime) ** 2))))

    dE = E_D_after - E_D_before
    rel_err = dE / E_D_before if E_D_before != 0.0 else float("nan")

    print("======================================================================")
    print("GAUGE SYMMETRY NUMERICAL CHECK (COVARIANT DIRAC ENERGY)")
    print("======================================================================")
    print(f"Dirac energy before gauge transform : E_D  = {E_D_before:.10e}")
    print(f"Dirac energy after  gauge transform : E_D' = {E_D_after:.10e}")
    print(f"Difference ΔE = E_D' - E_D           : ΔE  = {dE:.10e}")
    print(f"Relative error ΔE / E_D              :     = {rel_err:.10e}")
    print()
    print(f"Norm ||Psi|| before = {norm_before:.10e}")
    print(f"Norm ||Psi'|| after = {norm_after:.10e}")
    print("======================================================================")

    return {
        "E_D_before": E_D_before,
        "E_D_after": E_D_after,
        "delta_E": dE,
        "rel_error": rel_err,
        "norm_before": norm_before,
        "norm_after": norm_after,
    }


# =============================================================================
# SECTION 2: FINITE-HILBERT TWO-EXCITATION ANTISYMMETRY MODEL
# =============================================================================

@dataclass
class FiniteSubstrateParams:
    Lx: int = 2
    Ly: int = 2
    J_hop: float = 1.0
    m: float = 0.1
    g_defrag: float = 1.0
    sigma_defrag: float = 1.0
    lambda_G: float = 5.0
    lambda_S: float = -1.0
    lambda_T: float = 0.0
    J_exch: float = 1.0
    max_eigsh_iter: int = 5000


def site_index(x: int, y: int, Lx: int, Ly: int) -> int:
    return x + Lx * y


def site_coords(r: int, Lx: int, Ly: int):
    x = r % Lx
    y = r // Lx
    return x, y


def build_neighbors(Lx: int, Ly: int):
    Ns = Lx * Ly
    neighbors = {r: [] for r in range(Ns)}
    for y in range(Ly):
        for x in range(Lx):
            r = site_index(x, y, Lx, Ly)
            xp_ = (x + 1) % Lx
            xm_ = (x - 1) % Lx
            yp_ = (y + 1) % Ly
            ym_ = (y - 1) % Ly
            neighbors[r].append(site_index(xp_, y, Lx, Ly))
            neighbors[r].append(site_index(xm_, y, Lx, Ly))
            neighbors[r].append(site_index(x, yp_, Lx, Ly))
            neighbors[r].append(site_index(x, ym_, Lx, Ly))
    return neighbors


def defrag_potential(r: int, params: FiniteSubstrateParams) -> float:
    Lx, Ly = params.Lx, params.Ly
    x, y = site_coords(r, Lx, Ly)
    cx = 0.5 * (Lx - 1)
    cy = 0.5 * (Ly - 1)
    dx_ = x - cx
    dy_ = y - cy
    dist2 = dx_ * dx_ + dy_ * dy_
    if params.sigma_defrag <= 0.0:
        return 0.0
    return -np.exp(-dist2 / (2.0 * params.sigma_defrag ** 2))


def encode_basis(r1: int, s1: int, r2: int, s2: int, Ns: int) -> int:
    return (((r1 * 2 + s1) * Ns * 2) + (r2 * 2 + s2))


def decode_basis(idx: int, Ns: int):
    tmp = idx
    r2s2 = tmp % (Ns * 2)
    tmp //= (Ns * 2)
    r1s1 = tmp

    r1 = r1s1 // 2
    s1 = r1s1 % 2
    r2 = r2s2 // 2
    s2 = r2s2 % 2
    return r1, s1, r2, s2


def build_finite_substrate_hamiltonian(params: FiniteSubstrateParams) -> csr_matrix:
    Lx, Ly = params.Lx, params.Ly
    Ns = Lx * Ly
    dim = Ns * 2 * Ns * 2

    neighbors = build_neighbors(Lx, Ly)
    H = lil_matrix((dim, dim), dtype=np.complex128)

    V_defrag_site = np.array([defrag_potential(r, params) for r in range(Ns)], dtype=float)
    rho0 = 2.0 / Ns

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis(idx, Ns)

        # mass
        H[idx, idx] += 2.0 * params.m

        # defrag
        H[idx, idx] += params.g_defrag * (V_defrag_site[r1] + V_defrag_site[r2])

        # Gauss-like penalty
        occ = np.zeros(Ns, dtype=int)
        occ[r1] += 1
        occ[r2] += 1
        G = occ.astype(float) - rho0
        gauss_energy = 0.5 * params.lambda_G * np.sum(G * G)
        H[idx, idx] += gauss_energy

        # contact spin term at overlap
        if r1 == r2:
            sz1 = +0.5 if s1 == 0 else -0.5
            sz2 = +0.5 if s2 == 0 else -0.5

            H[idx, idx] += params.J_exch * (sz1 * sz2)
            if s1 == s2:
                H[idx, idx] += params.lambda_T

            if s1 != s2:
                H[idx, idx] += params.lambda_S
                s1p, s2p = s2, s1
                idx_flip = encode_basis(r1, s1p, r2, s2p, Ns)
                H[idx_flip, idx] += 0.5 * params.J_exch

        # hopping (particle 1)
        for r1p in neighbors[r1]:
            idx_new = encode_basis(r1p, s1, r2, s2, Ns)
            H[idx_new, idx] += -params.J_hop

        # hopping (particle 2)
        for r2p in neighbors[r2]:
            idx_new = encode_basis(r1, s1, r2p, s2, Ns)
            H[idx_new, idx] += -params.J_hop

    return H.tocsr()


def antisymmetry_metrics(psi: np.ndarray, params: FiniteSubstrateParams) -> Dict[str, float]:
    Lx, Ly = params.Lx, params.Ly
    Ns = Lx * Ly
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))
    total_norm = float(np.vdot(psi, psi).real)

    antisym_violation = 0.0
    sym_violation = 0.0

    for idx in range(dim):
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


def overlap_and_spin_metrics(psi: np.ndarray, params: FiniteSubstrateParams) -> Dict[str, float]:
    Lx, Ly = params.Lx, params.Ly
    Ns = Lx * Ly
    dim = Ns * 2 * Ns * 2

    psi = psi.reshape((dim,))

    overlap_prob = 0.0
    singlet_same_site = 0.0
    triplet_same_site = 0.0

    for r in range(Ns):
        amp_uu = psi[encode_basis(r, 0, r, 0, Ns)]
        amp_ud = psi[encode_basis(r, 0, r, 1, Ns)]
        amp_du = psi[encode_basis(r, 1, r, 0, Ns)]
        amp_dd = psi[encode_basis(r, 1, r, 1, Ns)]

        site_prob = (
            abs(amp_uu) ** 2
            + abs(amp_ud) ** 2
            + abs(amp_du) ** 2
            + abs(amp_dd) ** 2
        )
        overlap_prob += site_prob

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


def gauss_energy_expectation(psi: np.ndarray, params: FiniteSubstrateParams) -> float:
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


def run_finite_substrate_experiment(params: FiniteSubstrateParams) -> Dict[str, Any]:
    """
    Build the finite-Hilbert Hamiltonian, diagonalize the ground state,
    and compute antisymmetry + overlap diagnostics.
    """
    print("======================================================================")
    print("FINITE-HILBERT SUBSTRATE: TWO-EXCITATION SECTOR (substrate_engine)")
    print("======================================================================")
    print("Substrate parameters:")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("----------------------------------------------------------------------")

    H = build_finite_substrate_hamiltonian(params)
    dim = H.shape[0]
    print(f"[INFO] Hilbert dimension (two excitations) = {dim}")
    print("[INFO] Solving for ground state (smallest eigenvalue)...")

    evals, evecs = eigsh(H, k=1, which="SA", maxiter=params.max_eigsh_iter)
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    print(f"[RESULT] Ground state energy E0 = {E0:.6f}")

    norm = np.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    anti = antisymmetry_metrics(psi0, params)
    overlap = overlap_and_spin_metrics(psi0, params)
    E_gauss = gauss_energy_expectation(psi0, params)

    print("----------------------------------------------------------------------")
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
    print("----------------------------------------------------------------------")

    if anti["antisym_score"] > 0.95:
        print("[VERDICT] Ground state is strongly exchange-antisymmetric (fermion-like) "
              "within this toy model.")
    elif overlap["overlap_prob"] < 0.05 and overlap["singlet_fraction"] > 0.9:
        print("[VERDICT] Ground state strongly suppresses overlap and prefers singlets "
              "when overlap occurs (partial fermion-like behavior).")
    else:
        print("[VERDICT] No strong emergent antisymmetry in this parameter regime; "
              "behavior is mixed.")
    print("======================================================================")

    return {
        "E0": E0,
        "antisym_score": anti["antisym_score"],
        "sym_score": anti["sym_score"],
        "overlap_prob": overlap["overlap_prob"],
        "singlet_fraction": overlap["singlet_fraction"],
        "E_gauss": E_gauss,
    }


# =============================================================================
# SECTION 3: CHSH DIMER / BELL-TEST CORE
# =============================================================================

@dataclass
class CHSHParams:
    J: float = 1.0
    hz: float = 0.0
    t_max: float = 5.0
    n_steps: int = 50
    initial_state: str = "bell"  # "bell" or "product"


def build_chsh_operators():
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    I = qeye(2)

    # Measurement directions for maximal CHSH violation with the singlet:
    # a  = z, a' = x
    # b  = (z + x)/sqrt(2), b' = (z - x)/sqrt(2)
    a = sz
    ap = sx

    b = (sz + sx) / np.sqrt(2.0)
    bp = (sz - sx) / np.sqrt(2.0)

    A = tensor(a, I)
    Ap = tensor(ap, I)
    B = tensor(I, b)
    Bp = tensor(I, bp)

    B_chsh = A * B + A * Bp + Ap * B - Ap * Bp
    return B_chsh


def build_heisenberg_hamiltonian(J: float, hz: float):
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

    H_exch = J * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
    H_field = 0.5 * hz * (sz1 + sz2)
    return H_exch + H_field


def build_initial_state(initial_state: str):
    if initial_state == "bell":
        up = basis(2, 0)
        down = basis(2, 1)
        psi = (tensor(up, down) - tensor(down, up)).unit()  # singlet
        return psi
    else:
        up = basis(2, 0)
        return tensor(up, up)  # product state


def run_chsh_dimer_experiment(params: CHSHParams) -> Dict[str, Any]:
    """
    Heisenberg dimer with CHSH operator.
    Time-evolve and compute S(t) = <B_CHSH>_t.
    """
    print("======================================================================")
    print("CHSH DIMER EVOLUTION (substrate_engine)")
    print("======================================================================")
    print("CHSH parameters:")
    for k, v in asdict(params).items():
        print(f"  {k:15s} = {v}")
    print("----------------------------------------------------------------------")

    H = build_heisenberg_hamiltonian(params.J, params.hz)
    B_chsh = build_chsh_operators()
    psi0 = build_initial_state(params.initial_state)

    tlist = np.linspace(0.0, params.t_max, params.n_steps)
    result = mesolve(H, psi0, tlist, [], [])
    S_vals = np.array([expect(B_chsh, state) for state in result.states])

    max_idx = int(np.argmax(np.abs(S_vals)))
    S_max = float(S_vals[max_idx])
    t_max_S = float(tlist[max_idx])

    print("Time series (first few points):")
    print("   t        S(t)")
    print("  ----------------------")
    for i in range(min(6, len(tlist))):
        print(f"  {tlist[i]:6.4f}   {S_vals[i]:6.4f}")
    print("  ...")
    print("----------------------------------------------------------------------")
    print(f"Max |S(t)|: {abs(S_max):.6f} at t = {t_max_S:.6f}")
    print()
    print("Notes:")
    print("  - Local realistic theories satisfy |S| ≤ 2.")
    print("  - Quantum mechanics allows |S| up to 2√2 ≈ 2.828 (Tsirelson bound).")
    print("  - With the Bell (singlet) state and these measurement settings,")
    print("    you should see |S(t)| near 2.828 at some times (including t=0).")
    print("======================================================================")

    return {
        "t": tlist,
        "S_t": S_vals,
        "S_max": S_max,
        "t_at_S_max": t_max_S,
    }


# =============================================================================
# SECTION 4: CLI / MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="substrate_engine: unified Dirac+EM gauge lab, "
                    "finite-Hilbert antisymmetry, and CHSH dimer core."
    )

    # Which experiments to run
    p.add_argument("--run-dirac", action="store_true",
                   help="Run Dirac+EM gauge symmetry test.")
    p.add_argument("--no-run-dirac", action="store_true",
                   help="Disable Dirac+EM test.")
    p.add_argument("--run-finite", action="store_true",
                   help="Run finite-Hilbert antisymmetry model.")
    p.add_argument("--no-run-finite", action="store_true",
                   help="Disable finite-Hilbert model.")
    p.add_argument("--run-chsh", action="store_true",
                   help="Run CHSH dimer experiment.")
    p.add_argument("--no-run-chsh", action="store_true",
                   help="Disable CHSH dimer experiment.")

    # Dirac+EM parameters
    p.add_argument("--Nx", type=int, default=32, help="Grid size in x.")
    p.add_argument("--Ny", type=int, default=32, help="Grid size in y.")
    p.add_argument("--dx", type=float, default=1.0, help="Spatial grid spacing.")
    p.add_argument("--dt", type=float, default=0.01, help="Time step.")
    p.add_argument("--steps", type=int, default=200, help="Number of warmup steps.")
    p.add_argument("--c", type=float, default=1.0, help="Signal speed (c=1 units).")
    p.add_argument("--q", type=float, default=1.0, help="Charge of Dirac field.")
    p.add_argument("--m", type=float, default=0.1, help="Dirac mass.")
    p.add_argument("--defrag", action="store_true",
                   help="Enable defrag scalar field phi.")
    p.add_argument("--defrag-kappa", type=float, default=0.1,
                   help="Defrag relaxation rate.")
    p.add_argument("--defrag-lambda", type=float, default=1.0,
                   help="Defrag Laplacian weight.")
    p.add_argument("--defrag-g", type=float, default=0.5,
                   help="Coupling strength of defrag field to mass term.")
    p.add_argument("--chi-amp", type=float, default=0.5,
                   help="Amplitude of gauge phase χ(x,y).")
    p.add_argument("--chi-mode-x", type=int, default=1,
                   help="Mode number in x for χ(x,y).")
    p.add_argument("--chi-mode-y", type=int, default=1,
                   help="Mode number in y for χ(x,y).")
    p.add_argument("--out-dir", type=str, default="gauge_lab_output",
                   help="Directory for gauge lab outputs.")

    # Finite-Hilbert substrate parameters
    p.add_argument("--Lx", type=int, default=2, help="Lattice size in x (finite model).")
    p.add_argument("--Ly", type=int, default=2, help="Lattice size in y (finite model).")
    p.add_argument("--J-hop", type=float, default=1.0, dest="J_hop",
                   help="Hopping strength J_hop (finite model).")
    p.add_argument("--mass", type=float, default=0.1,
                   help="Mass term per excitation (finite model).")
    p.add_argument("--g-defrag", type=float, default=1.0,
                   help="Defrag strength g_defrag (finite model).")
    p.add_argument("--sigma-defrag", type=float, default=1.0,
                   help="Defrag Gaussian width (finite model).")
    p.add_argument("--lambda-G", type=float, default=5.0, dest="lambda_G",
                   help="Gauss-like penalty strength (finite model).")
    p.add_argument("--lambda-S", type=float, default=-1.0, dest="lambda_S",
                   help="Singlet bonus at overlap (finite model).")
    p.add_argument("--lambda-T", type=float, default=0.0, dest="lambda_T",
                   help="Triplet penalty at overlap (finite model).")
    p.add_argument("--J-exch", type=float, default=1.0, dest="J_exch",
                   help="Exchange strength at overlap (finite model).")
    p.add_argument("--max-eigsh-iter", type=int, default=5000,
                   help="Maximum iterations for eigsh diagonalization.")

    # CHSH parameters
    p.add_argument("--chsh-J", type=float, default=1.0,
                   help="Heisenberg coupling J for CHSH dimer.")
    p.add_argument("--chsh-hz", type=float, default=0.0,
                   help="Longitudinal field hz for CHSH dimer.")
    p.add_argument("--chsh-t-max", type=float, default=5.0,
                   help="Maximum time for CHSH evolution.")
    p.add_argument("--chsh-n-steps", type=int, default=50,
                   help="Number of time steps for CHSH evolution.")
    p.add_argument("--chsh-initial", type=str, default="bell",
                   help="Initial state for CHSH dimer ('bell' or 'product').")

    return p.parse_args()


def main():
    args = parse_args()

    # defaults: run all three unless explicitly disabled
    run_dirac = True
    run_finite = True
    run_chsh = True

    if args.run_dirac:
        run_dirac = True
    if args.no_run_dirac:
        run_dirac = False

    if args.run_finite:
        run_finite = True
    if args.no_run_finite:
        run_finite = False

    if args.run_chsh:
        run_chsh = True
    if args.no_run_chsh:
        run_chsh = False

    if run_dirac:
        dparams = DiracGaugeParams(
            Nx=args.Nx,
            Ny=args.Ny,
            dx=args.dx,
            dt=args.dt,
            steps=args.steps,
            c=args.c,
            q=args.q,
            m=args.m,
            defrag=args.defrag,
            defrag_kappa=args.defrag_kappa,
            defrag_lambda=args.defrag_lambda,
            defrag_g=args.defrag_g,
            chi_amp=args.chi_amp,
            chi_mode_x=args.chi_mode_x,
            chi_mode_y=args.chi_mode_y,
            out_dir=args.out_dir,
        )
        run_dirac_gauge_experiment(dparams)

    if run_finite:
        fparams = FiniteSubstrateParams(
            Lx=args.Lx,
            Ly=args.Ly,
            J_hop=args.J_hop,
            m=args.mass,
            g_defrag=args.g_defrag,
            sigma_defrag=args.sigma_defrag,
            lambda_G=args.lambda_G,
            lambda_S=args.lambda_S,
            lambda_T=args.lambda_T,
            J_exch=args.J_exch,
            max_eigsh_iter=args.max_eigsh_iter,
        )
        run_finite_substrate_experiment(fparams)

    if run_chsh:
        cparams = CHSHParams(
            J=args.chsh_J,
            hz=args.chsh_hz,
            t_max=args.chsh_t_max,
            n_steps=args.chsh_n_steps,
            initial_state=args.chsh_initial,
        )
        run_chsh_dimer_experiment(cparams)


if __name__ == "__main__":
    main()
