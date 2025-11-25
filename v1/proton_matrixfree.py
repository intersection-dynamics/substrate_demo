#!/usr/bin/env python3
"""
proton_matrixfree.py

Matrix-free 3-quark "proton" simulator on a 3D lattice.

Key idea:
    - Do NOT build the Hamiltonian H explicitly.
    - Represent the full 3-quark wavefunction ψ as a tensor
        ψ[r1, s1, r2, s2, r3, s3]
      with r_i in {0..Ns-1}, s_i in {0,1} for ↑/↓.
    - Implement H·ψ directly using vectorized NumPy operations.
    - Wrap H as a scipy.sparse.linalg.LinearOperator and call eigsh.

Hamiltonian (simplified):

    H = H_hop + H_mass + H_defrag + H_Gauss + H_Zeeman

We intentionally drop the spin-contact/exchange terms here to keep the
matrix-free matvec manageable at large Hilbert dimension.

    - H_hop:
        nearest-neighbor hopping J_hop for each quark (periodic BCs).
    - H_mass:
        3 * m on every configuration.
    - H_defrag:
        g_defrag * sum_i V_defrag(r_i)
        where V_defrag is a 3D Gaussian centered in the box.
    - H_Gauss:
        0.5 * lambda_G * sum_r (occ(r) - rho0)^2
        where occ(r) is the number of quarks at site r (0..3),
        and rho0 = 3 / Ns.
    - H_Zeeman:
        -B * (S_z1 + S_z2 + S_z3) with S_z = ±1/2.

Usage (from substrate_demo folder):

    python proton_matrixfree.py --Lx 3 --Ly 3 --Lz 3 --g_defrag 2.0 --sigma_defrag 0.7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

from substrate_engine_3d import site_coords_3d, build_neighbors_3d


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class ThreeQuarkMFParams:
    # Lattice
    Lx: int = 2
    Ly: int = 2
    Lz: int = 2

    # Substrate / single-particle
    J_hop: float = 1.0
    m: float = 0.1
    g_defrag: float = 1.0
    sigma_defrag: float = 1.0

    # Gauss-like constraint
    lambda_G: float = 5.0

    # Zeeman field (select S_z)
    B_field: float = 0.0

    # Solver
    max_eigsh_iter: int = 500
    k_eigs: int = 1

    # Random seed for reproducibility
    seed: int = 42


# =============================================================================
# Helper: defrag potential and Gauss energy table
# =============================================================================

def defrag_potential_3d(params: ThreeQuarkMFParams) -> np.ndarray:
    """
    3D Gaussian defrag potential defined on sites r=0..Ns-1,
    using (x,y,z) from site_coords_3d.
    V_defrag(r) = -exp(-|r - r0|^2 / (2 sigma^2))
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


def gauss_energy_table(params: ThreeQuarkMFParams) -> np.ndarray:
    """
    Precompute a 3D table G[r1, r2, r3] for the Gauss term:

        E_Gauss = 0.5 * lambda_G * sum_r (occ(r) - rho0)^2

    where occ(r) is the number of quarks at site r (0..3),
    and rho0 = 3/Ns.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    rho0 = 3.0 / Ns

    G = np.zeros((Ns, Ns, Ns), dtype=float)

    for r1 in range(Ns):
        for r2 in range(Ns):
            for r3 in range(Ns):
                occ = np.zeros(Ns, dtype=int)
                occ[r1] += 1
                occ[r2] += 1
                occ[r3] += 1
                diff = occ.astype(float) - rho0
                G[r1, r2, r3] = 0.5 * params.lambda_G * np.sum(diff * diff)

    return G


# =============================================================================
# Matrix-free Hamiltonian (H·psi)
# =============================================================================

class ThreeQuarkHamiltonianMF:
    """
    Matrix-free 3-quark Hamiltonian:

        H = H_hop + H_mass + H_defrag + H_Gauss + H_Zeeman

    acting on ψ[r1,s1,r2,s2,r3,s3].
    """

    def __init__(self, params: ThreeQuarkMFParams):
        self.params = params
        self.Lx = params.Lx
        self.Ly = params.Ly
        self.Lz = params.Lz
        self.Ns = self.Lx * self.Ly * self.Lz

        # neighbor dict {r: [neighbors...]} and array [Ns, num_neighbors]
        self.neighbors = build_neighbors_3d(self.Lx, self.Ly, self.Lz)
        self.neigh_arr = np.stack(
            [np.array(self.neighbors[r], dtype=int) for r in range(self.Ns)],
            axis=0,
        )  # shape [Ns, num_neighbors]

        # defrag potential per site
        self.V_defrag = defrag_potential_3d(params)

        # Gauss energy table for triples (r1, r2, r3)
        self.G_table = gauss_energy_table(params)

        # for convenience
        self.dim = (2 * self.Ns) ** 3

    # -- utility reshape helpers --

    def flat_to_tensor(self, psi_flat: np.ndarray) -> np.ndarray:
        """
        Reshape flat vector of length dim to ψ[r1,s1,r2,s2,r3,s3].
        """
        Ns = self.Ns
        return psi_flat.reshape(Ns, 2, Ns, 2, Ns, 2)

    def tensor_to_flat(self, psi_tensor: np.ndarray) -> np.ndarray:
        return psi_tensor.reshape(-1)

    # -- H·ψ implementation --

    def apply(self, psi_flat: np.ndarray) -> np.ndarray:
        """
        Compute H·psi in a matrix-free manner.
        """
        p = self.params
        Ns = self.Ns

        psi = self.flat_to_tensor(psi_flat)

        # output accumulator
        Hpsi = np.zeros_like(psi, dtype=np.complex128)

        # 1) Mass term: 3m * ψ
        Hpsi += 3.0 * p.m * psi

        # 2) Defrag potential: g_defrag * sum_i V_defrag[r_i]
        V1 = self.V_defrag  # length Ns
        V_sum = (
            V1.reshape(Ns, 1, 1)
            + V1.reshape(1, Ns, 1)
            + V1.reshape(1, 1, Ns)
        )
        Hpsi += p.g_defrag * V_sum[:, None, :, None, :, None] * psi

        # 3) Gauss term: G_table[r1,r2,r3]
        Hpsi += self.G_table[:, None, :, None, :, None] * psi

        # 4) Zeeman term: -B * (S_z1 + S_z2 + S_z3)
        if p.B_field != 0.0:
            sz = np.array([+0.5, -0.5], dtype=float)
            S1 = sz.reshape(1, 2, 1, 1, 1, 1)
            S2 = sz.reshape(1, 1, 1, 2, 1, 1)
            S3 = sz.reshape(1, 1, 1, 1, 1, 2)
            S_total = S1 + S2 + S3
            Hpsi += -p.B_field * S_total * psi

        # 5) Hopping term: -J_hop * sum_neighbors ψ
        if p.J_hop != 0.0:
            J = p.J_hop
            neigh_arr = self.neigh_arr  # [Ns, num_neighbors]

            # particle 1 hops: axis 0 (r1)
            psi_reshaped = psi.reshape(Ns, -1)              # [Ns, rest]
            psi_neigh = psi_reshaped[neigh_arr]             # [Ns, num_neighbors, rest]
            psi1hop = psi_neigh.sum(axis=1).reshape(psi.shape)

            # particle 2 hops: axis 2 (r2)
            psi_trans = np.transpose(psi, (2, 3, 0, 1, 4, 5))   # r2 axis 0
            psi2_reshaped = psi_trans.reshape(Ns, -1)
            psi2_neigh = psi2_reshaped[neigh_arr]
            psi2hop = psi2_neigh.sum(axis=1).reshape(psi_trans.shape)
            psi2hop = np.transpose(psi2hop, (2, 3, 0, 1, 4, 5))

            # particle 3 hops: axis 4 (r3)
            psi_trans3 = np.transpose(psi, (4, 5, 0, 1, 2, 3))  # r3 axis 0
            psi3_reshaped = psi_trans3.reshape(Ns, -1)
            psi3_neigh = psi3_reshaped[neigh_arr]
            psi3hop = psi3_neigh.sum(axis=1).reshape(psi_trans3.shape)
            psi3hop = np.transpose(psi3hop, (2, 3, 4, 5, 0, 1))

            Hpsi += -J * (psi1hop + psi2hop + psi3hop)

        return self.tensor_to_flat(Hpsi)


# =============================================================================
# Observables
# =============================================================================

def analyze_state(
    params: ThreeQuarkMFParams,
    psi_flat: np.ndarray
) -> Dict[str, Any]:
    """
    Compute per-site occupancy, barycenter, effective proton radius,
    and S_z statistics from the ground-state wavefunction.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz

    psi = psi_flat.reshape(Ns, 2, Ns, 2, Ns, 2)

    prob = np.abs(psi) ** 2
    Z = prob.sum()
    if Z <= 0:
        Z = 1.0
    prob /= Z

    occ = np.zeros(Ns, dtype=float)
    pos_weighted = np.zeros(3, dtype=float)
    total_quark_weight = 0.0

    sz_vals = np.array([+0.5, -0.5], dtype=float)
    sz_tot_exp = 0.0
    sz_tot_sq_exp = 0.0
    sz_hist = {1.5: 0.0, 0.5: 0.0, -0.5: 0.0, -1.5: 0.0}

    # Iterate over positions only; sum over spins vectorized
    for r1 in range(Ns):
        x1, y1, z1 = site_coords_3d(r1, Lx, Ly, Lz)
        for r2 in range(Ns):
            x2, y2, z2 = site_coords_3d(r2, Lx, Ly, Lz)
            for r3 in range(Ns):
                x3, y3, z3 = site_coords_3d(r3, Lx, Ly, Lz)

                block = prob[r1, :, r2, :, r3, :]  # (2,2,2)
                p_block = block.sum()
                if p_block < 1e-16:
                    continue

                # occupancies & barycenter
                for r, (x, y, z) in zip(
                    (r1, r2, r3),
                    ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3))
                ):
                    occ[r] += p_block
                    pos_weighted += p_block * np.array([x, y, z], dtype=float)
                    total_quark_weight += p_block

                # exact S_z and S_z^2 over spins
                for s1 in (0, 1):
                    for s2 in (0, 1):
                        for s3 in (0, 1):
                            p_cfg = block[s1, s2, s3]
                            if p_cfg < 1e-16:
                                continue
                            sz1 = sz_vals[s1]
                            sz2 = sz_vals[s2]
                            sz3 = sz_vals[s3]
                            sz_tot = sz1 + sz2 + sz3
                            sz_tot_exp += p_cfg * sz_tot
                            sz_tot_sq_exp += p_cfg * (sz_tot ** 2)
                            if sz_tot in sz_hist:
                                sz_hist[sz_tot] += p_cfg

    # normalize occ to 3 quarks total
    occ_sum = occ.sum()
    if occ_sum > 0:
        occ *= (3.0 / occ_sum)

    # barycenter
    if total_quark_weight > 0:
        barycenter = pos_weighted / (3.0 * total_quark_weight)
    else:
        barycenter = np.array([0.0, 0.0, 0.0], dtype=float)

    # proton radius: <r^2> relative to barycenter
    r2_acc = 0.0
    for r in range(Ns):
        x, y, z = site_coords_3d(r, Lx, Ly, Lz)
        dx, dy, dz = x - barycenter[0], y - barycenter[1], z - barycenter[2]
        r2 = dx * dx + dy * dy + dz * dz
        r2_acc += occ[r] * r2
    r2_per_quark = r2_acc / 3.0
    r_eff = np.sqrt(r2_per_quark)

    sz_hist = {float(k): float(v) for k, v in sz_hist.items()}

    return {
        "occ_per_site": occ.tolist(),
        "barycenter": barycenter.tolist(),
        "effective_proton_radius": float(r_eff),
        "S_z_expectation": float(sz_tot_exp),
        "S_z2_expectation": float(sz_tot_sq_exp),
        "S_z_histogram": sz_hist,
        "dim": int(psi.size),
    }


# =============================================================================
# Main / CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Matrix-free 3-quark proton simulator on a 3D lattice."
    )
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=2)
    parser.add_argument("--Lz", type=int, default=2)
    parser.add_argument("--J_hop", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--g_defrag", type=float, default=1.0)
    parser.add_argument("--sigma_defrag", type=float, default=1.0)
    parser.add_argument("--lambda_G", type=float, default=5.0)
    parser.add_argument("--B", type=float, default=0.0)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = ThreeQuarkMFParams(
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        J_hop=args.J_hop,
        m=args.m,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        lambda_G=args.lambda_G,
        B_field=args.B,
        max_eigsh_iter=args.max_iter,
        seed=args.seed,
    )

    np.random.seed(params.seed)

    print("\n" + "=" * 70)
    print("MATRIX-FREE 3-QUARK PROTON SIMULATOR")
    print("=" * 70)
    print("Parameters:")
    for k, v in asdict(params).items():
        print(f"  {k:14s} = {v}")
    print("=" * 70)

    Hmf = ThreeQuarkHamiltonianMF(params)
    dim = Hmf.dim
    print(f"Effective Hilbert dimension dim = {dim}")

    # define LinearOperator for eigsh
    def matvec(v: np.ndarray) -> np.ndarray:
        return Hmf.apply(v)

    H_linop = LinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        dtype=np.complex128,
    )

    print("\nRunning eigsh (matrix-free)...")
    evals, evecs = eigsh(
        H_linop,
        k=params.k_eigs,
        which="SA",
        maxiter=params.max_eigsh_iter,
    )
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    print(f"Ground state energy E0 = {E0:.6f}")

    # Analyze observables
    obs = analyze_state(params, psi0)
    print("\nGround state observables:")
    print(f"  dim               = {obs['dim']}")
    print(f"  r_eff (radius)    = {obs['effective_proton_radius']:.6f}")
    print(f"  <S_z>             = {obs['S_z_expectation']:.6f}")
    print(f"  <S_z^2>           = {obs['S_z2_expectation']:.6f}")
    print("  S_z histogram     =")
    for sz, p in sorted(obs["S_z_histogram"].items(), reverse=True):
        print(f"    S_z = {sz:+.1f} : {p:.4f}")

    bx, by, bz = obs["barycenter"]
    print(f"\nBarycenter (x,y,z): ({bx:.3f}, {by:.3f}, {bz:.3f})")

    print("\nSite occupancy (normalized to 3 quarks total):")
    occ = np.array(obs["occ_per_site"])
    for r in range(Hmf.Ns):
        x, y, z = site_coords_3d(r, params.Lx, params.Ly, params.Lz)
        print(f"  r={r:3d} (x,y,z)=({x},{y},{z}) : n(r) ≈ {occ[r]:.6f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
