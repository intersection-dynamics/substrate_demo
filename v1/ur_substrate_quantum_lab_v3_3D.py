#!/usr/bin/env python3
"""
ur_substrate_quantum_lab_v3_3D.py

Framework 1.3 Implementation:
A 3D Finite-Hilbert Substrate with Explicit 2-Particle Subspace Projection.

This script tests if the antisymmetry (fermion-like behavior) observed in 
the 2D V2 model persists in a 3D cubic lattice geometry.

Key Features:
1. 3D Geometry: Lx * Ly * Lz lattice.
2. Subspace Projection: Constructs H only in the N=2 sector (drastic speedup).
3. Exchange Diagnostics: Computes Symmetry (S) vs Antisymmetry (A) scores.
"""

import argparse
import math
import itertools
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass, asdict

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SubstrateParams3D:
    Lx: int = 2
    Ly: int = 2
    Lz: int = 2          # New: Z-dimension
    J_hop: float = 1.0
    m: float = 0.1
    g_defrag: float = 1.0
    sigma_defrag: float = 1.0
    lambda_G: float = 10.0   # Increased for 3D stability
    lambda_S: float = -2.0   # Stronger singlet bias
    lambda_T: float = 2.0    # Stronger triplet penalty
    J_exch: float = 1.0
    pbc: bool = True         # Periodic Boundary Conditions

# =============================================================================
# BASIS CONSTRUCTION (N=2 SUBSPACE)
# =============================================================================

class BasisManager:
    def __init__(self, params: SubstrateParams3D):
        self.p = params
        self.Ns = params.Lx * params.Ly * params.Lz
        
        # 1. Generate all valid single-particle states |r, s>
        # r in [0, Ns-1], s in [0, 1] (up/down)
        self.single_states = []
        for r in range(self.Ns):
            for s in [0, 1]:
                self.single_states.append((r, s))
        
        self.dim_1 = len(self.single_states) # 2 * Ns
        
        # 2. Generate all 2-particle basis states |r1, s1; r2, s2>
        # We do not enforce ordering here to allow full Hilbert space exploration
        # (distinguishable particles), but we map them to a linear index.
        self.basis_map = {} # (r1,s1,r2,s2) -> index
        self.idx_map = []   # index -> (r1,s1,r2,s2)
        
        idx = 0
        for i in range(self.dim_1):
            for j in range(self.dim_1):
                r1, s1 = self.single_states[i]
                r2, s2 = self.single_states[j]
                
                state = (r1, s1, r2, s2)
                self.basis_map[state] = idx
                self.idx_map.append(state)
                idx += 1
                
        self.dim = len(self.idx_map)
        print(f"[INIT] 3D Lattice {params.Lx}x{params.Ly}x{params.Lz} (Ns={self.Ns})")
        print(f"[INIT] 2-Particle Subspace Dimension: {self.dim}")

    def get_idx(self, r1, s1, r2, s2):
        return self.basis_map.get((r1, s1, r2, s2))

    def get_state(self, idx):
        return self.idx_map[idx]

# =============================================================================
# GEOMETRY & POTENTIALS
# =============================================================================

def get_coords(r, Lx, Ly, Lz):
    z = r // (Lx * Ly)
    rem = r % (Lx * Ly)
    y = rem // Lx
    x = rem % Lx
    return x, y, z

def get_r(x, y, z, Lx, Ly, Lz):
    return x + Lx * (y + Ly * z)

def get_neighbors(r, p: SubstrateParams3D):
    x, y, z = get_coords(r, p.Lx, p.Ly, p.Lz)
    neighbors = []
    
    shifts = [
        (1,0,0), (-1,0,0),
        (0,1,0), (0,-1,0),
        (0,0,1), (0,0,-1)
    ]
    
    for dx, dy, dz in shifts:
        nx, ny, nz = x+dx, y+dy, z+dz
        
        if p.pbc:
            nx %= p.Lx
            ny %= p.Ly
            nz %= p.Lz
        else:
            if not (0 <= nx < p.Lx and 0 <= ny < p.Ly and 0 <= nz < p.Lz):
                continue
                
        neighbors.append(get_r(nx, ny, nz, p.Lx, p.Ly, p.Lz))
        
    return neighbors

def defrag_val(r, p: SubstrateParams3D):
    if p.g_defrag == 0: return 0.0
    x, y, z = get_coords(r, p.Lx, p.Ly, p.Lz)
    # Center
    cx, cy, cz = (p.Lx-1)/2, (p.Ly-1)/2, (p.Lz-1)/2
    dist2 = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
    return -p.g_defrag * math.exp(-dist2 / (2 * p.sigma_defrag**2))

# =============================================================================
# HAMILTONIAN
# =============================================================================

def build_hamiltonian_3d(bm: BasisManager):
    p = bm.p
    dim = bm.dim
    H = lil_matrix((dim, dim), dtype=complex)
    
    # Precompute site potentials
    V_site = [defrag_val(r, p) for r in range(bm.Ns)]
    
    # Iterate through all basis states |u>
    for u_idx in range(dim):
        r1, s1, r2, s2 = bm.get_state(u_idx)
        
        # 1. Diagonal Terms
        diag_val = 0.0
        
        # Mass
        diag_val += 2.0 * p.m
        
        # Defrag
        diag_val += V_site[r1] + V_site[r2]
        
        # Gauss / Contact terms (active if r1 approx r2)
        # In this V2/V3 logic, we treat Gauss as an energy penalty for local density
        # rho0 = 2 / Ns. Occ(r) is 0, 1, or 2.
        
        occ = {}
        occ[r1] = occ.get(r1, 0) + 1
        occ[r2] = occ.get(r2, 0) + 1
        
        # Gauss Energy: lambda_G/2 * sum (n_r - rho0)^2
        # Constant shift irrelevant, we care about fluctuations.
        # If r1 == r2: n_r=2 at one site. If r1 != r2: n_r=1 at two sites.
        if r1 == r2:
            # Double occupancy penalty dominates
            # (2 - rho)^2 + (Ns-1)*rho^2 vs 2*(1-rho)^2 + ...
            # Simplified: Penalty for stacking
            diag_val += p.lambda_G 
            
            # Spin Contact Terms
            # s=0 (up), s=1 (down)
            sz1 = 0.5 if s1==0 else -0.5
            sz2 = 0.5 if s2==0 else -0.5
            
            # Heisenberg Diagonal: Sz * Sz
            diag_val += p.J_exch * (sz1 * sz2)
            
            if s1 == s2: # Triplet penalty (Parallel spins on same site)
                diag_val += p.lambda_T
            else:        # Singlet bonus (Opposite spins on same site)
                diag_val += p.lambda_S
        
        H[u_idx, u_idx] += diag_val
        
        # 2. Off-Diagonal Terms
        
        # Hopping (Particle 1)
        for nr1 in get_neighbors(r1, p):
            v_idx = bm.get_idx(nr1, s1, r2, s2)
            if v_idx is not None:
                H[v_idx, u_idx] -= p.J_hop
                
        # Hopping (Particle 2)
        for nr2 in get_neighbors(r2, p):
            v_idx = bm.get_idx(r1, s1, nr2, s2)
            if v_idx is not None:
                H[v_idx, u_idx] -= p.J_hop
                
        # Heisenberg Flip (S+ S- + S- S+) at same site
        if r1 == r2 and s1 != s2:
            # Flip spins: s1->s2, s2->s1
            v_idx = bm.get_idx(r1, s2, r2, s1)
            if v_idx is not None:
                H[v_idx, u_idx] += 0.5 * p.J_exch

    return H.tocsr()

# =============================================================================
# DIAGNOSTICS
# =============================================================================

def analyze_state(psi, bm: BasisManager):
    # 1. Exchange Symmetry
    # Swap particle 1 and 2: |r1, s1; r2, s2> -> |r2, s2; r1, s1>
    
    overlap_swap = 0.0 + 0j
    
    for idx in range(bm.dim):
        coeff = psi[idx]
        if abs(coeff) < 1e-15: continue
        
        r1, s1, r2, s2 = bm.get_state(idx)
        swapped_idx = bm.get_idx(r2, s2, r1, s1)
        
        # Inner product <Psi | P_ex | Psi>
        # Contribution is conj(psi[idx]) * psi[swapped_idx]
        overlap_swap += np.conj(coeff) * psi[swapped_idx]
        
    expectation_Pex = overlap_swap.real
    
    # A = 1 - <P_ex> (if normalized, ranges 0 to 2, usually normalized to 0..1)
    # Actually, let's use the metrics from V2:
    # Antisym A = 1 - || psi + P psi ||^2 / ... 
    # Simplifies to (1 - <P>)/2 for normalized states? 
    # Let's stick to V2 definition:
    
    # V2: Antisym score A = 1 - || psi + P psi ||^2 / 2 (assuming norm=1)
    # || psi + P psi ||^2 = <psi|psi> + <Ppsi|Ppsi> + <psi|Ppsi> + <Ppsi|psi>
    #                     = 2 + 2 <psi|Ppsi>
    # So A = 1 - (2 + 2<P>)/4 ? No, V2 def was unnormalized.
    
    # Let's just use <P_ex>:
    # If Antisymmetric: P|psi> = -|psi>, so <P> = -1.
    # If Symmetric:     P|psi> = +|psi>, so <P> = +1.
    
    antisym_score = (1.0 - expectation_Pex) / 2.0  # 1.0 for antisym, 0.0 for sym
    
    return {
        "P_ex_expectation": expectation_Pex,
        "Antisym_Score_Norm": antisym_score
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Lz", type=int, default=2, help="Z dimension")
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=2)
    args = parser.parse_args()
    
    params = SubstrateParams3D(Lx=args.Lx, Ly=args.Ly, Lz=args.Lz)
    
    print("="*60)
    print(f"FRAMEWORK 1.3: 3D SUBSTRATE ANALYSIS ({params.Lx}x{params.Ly}x{params.Lz})")
    print("="*60)
    
    bm = BasisManager(params)
    H = build_hamiltonian_3d(bm)
    
    print("[RUN] Diagonalizing...")
    evals, evecs = eigsh(H, k=1, which='SA')
    
    E0 = evals[0]
    psi0 = evecs[:, 0]
    
    # Normalize
    psi0 = psi0 / np.linalg.norm(psi0)
    
    print(f"[RESULT] Ground State Energy: {E0:.6f}")
    
    metrics = analyze_state(psi0, bm)
    P_ex = metrics["P_ex_expectation"]
    
    print("-" * 60)
    print(f"Exchange Expectation <P_ex>: {P_ex:.6f}")
    print(f"  Expected: +1.0 (Boson), -1.0 (Fermion)")
    print("-" * 60)
    
    if P_ex < -0.9:
        print(">> SUCCESS: Emergent Antisymmetry confirmed in 3D.")
    elif P_ex > 0.9:
        print(">> RESULT: Symmetric (Bosonic) Ground State.")
    else:
        print(">> RESULT: Mixed Symmetry State.")

if __name__ == "__main__":
    main()