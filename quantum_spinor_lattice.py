#!/usr/bin/env python
"""
quantum_spinor_lattice.py

Quantum lattice where each spatial site has a SPINOR degree of freedom.
Tests whether spinor structure (not just winding) produces fermion/boson distinction.

Key difference from previous code:
- Each site has spin-1/2 (two components: |↑⟩ and |↓⟩)
- Hamiltonian includes spin-dependent interactions
- Patterns can have non-trivial spin texture (not just spatial winding)

Hilbert space structure:
  H = ⊗_{spatial sites} (H_spatial ⊗ H_spin)
  
For 2×2 lattice with spin-1/2:
  - 4 spatial sites
  - Each site has 2 spin states
  - Total dimension: (2×2)^4 = 256 (tractable)
"""

import numpy as np
import argparse
import os
import json

try:
    from qutip import (Qobj, tensor, basis, qeye, sigmax, sigmay, sigmaz,
                       mesolve, entropy_vn)
    QUTIP_AVAILABLE = True
except ImportError:
    print("[ERROR] QuTiP required")
    import sys
    sys.exit(1)

class SpinorLattice:
    """
    Quantum lattice where each site has spatial + spin degrees of freedom.
    """
    
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.n_sites = Nx * Ny
        
        # Each site has 4 states: 2 (spatial/field occupation) × 2 (spin)
        # For simplicity: |0⟩,|1⟩ for field, |↑⟩,|↓⟩ for spin
        # Combined: |0↑⟩, |0↓⟩, |1↑⟩, |1↓⟩
        self.site_dim = 4
        
        self.total_dim = self.site_dim ** self.n_sites
        
        if self.total_dim > 2**20:
            print(f"[WARNING] Hilbert space dimension = {self.total_dim}")
            print(f"[WARNING] This may be too large")
        
        self.H = None
        self.state = None
        
        print(f"[INIT] Spinor lattice: {Nx}×{Ny} sites")
        print(f"[INIT] Each site: 2 (field) × 2 (spin) = 4 states")
        print(f"[INIT] Total Hilbert space: {self.site_dim}^{self.n_sites} = {self.total_dim}")
    
    def site_index(self, i, j):
        """2D to 1D index."""
        return i * self.Ny + j
    
    def build_operators(self):
        """Build single-site operators."""
        # Field operators (occupation number)
        # |0⟩ = empty, |1⟩ = occupied
        n_field = Qobj([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])  # Projects onto |1↑⟩ or |1↓⟩
        
        # Spin operators (act on spin subspace)
        # States: |0↑⟩=0, |0↓⟩=1, |1↑⟩=2, |1↓⟩=3
        sx = Qobj([[0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]]) / 2  # σx acts on spin
        
        sy = Qobj([[0, -1j, 0, 0],
                   [1j, 0, 0, 0],
                   [0, 0, 0, -1j],
                   [0, 0, 1j, 0]]) / 2  # σy acts on spin
        
        sz = Qobj([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]]) / 2  # σz acts on spin
        
        # Creation/annihilation for field
        a = Qobj([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0]])  # Lowers field occupation
        
        a_dag = a.dag()
        
        return n_field, sx, sy, sz, a, a_dag
    
    def single_site_op(self, op, site):
        """Embed operator at given site."""
        ops = [qeye(self.site_dim) for _ in range(self.n_sites)]
        ops[site] = op
        return tensor(ops)
    
    def two_site_op(self, op1, site1, op2, site2):
        """Two-site operator."""
        ops = [qeye(self.site_dim) for _ in range(self.n_sites)]
        ops[site1] = op1
        ops[site2] = op2
        return tensor(ops)
    
    def build_hamiltonian(self, J_hop=1.0, J_spin=0.5, g_spinorbit=0.2, mass=0.1):
        """
        Build Hamiltonian with:
        1. Nearest-neighbor hopping (field exchange)
        2. Spin-spin interaction
        3. Spin-orbit coupling (crucial for spinors!)
        4. Mass term
        """
        print("[BUILD] Constructing spinor Hamiltonian...")
        
        n_field, sx, sy, sz, a, a_dag = self.build_operators()
        
        # Start with zero
        H = 0 * tensor([qeye(self.site_dim) for _ in range(self.n_sites)])
        
        print("[BUILD] Adding hopping terms...")
        # Hopping: a†_i a_j (field hops between sites)
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                neighbors = [
                    self.site_index((i+1) % self.Nx, j),
                    self.site_index(i, (j+1) % self.Ny),
                ]
                
                for nb in neighbors:
                    H -= J_hop * self.two_site_op(a_dag, site, a, nb)
                    H -= J_hop * self.two_site_op(a, site, a_dag, nb)
        
        print("[BUILD] Adding spin interactions...")
        # Spin-spin coupling (Heisenberg-like)
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                neighbors = [
                    self.site_index((i+1) % self.Nx, j),
                    self.site_index(i, (j+1) % self.Ny),
                ]
                
                for nb in neighbors:
                    # S_i · S_j
                    H += J_spin * self.two_site_op(sx, site, sx, nb)
                    H += J_spin * self.two_site_op(sy, site, sy, nb)
                    H += J_spin * self.two_site_op(sz, site, sz, nb)
        
        print("[BUILD] Adding spin-orbit coupling...")
        # Spin-orbit: couples spin to spatial derivatives
        # When particle hops, spin rotates based on direction
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                # Hop right: spin rotates around z
                site_right = self.site_index((i+1) % self.Nx, j)
                H -= 1j * g_spinorbit * self.two_site_op(a_dag*sz, site, a, site_right)
                H += 1j * g_spinorbit * self.two_site_op(a, site, a_dag*sz, site_right)
                
                # Hop up: spin rotates differently
                site_up = self.site_index(i, (j+1) % self.Ny)
                H -= 1j * g_spinorbit * self.two_site_op(a_dag*sx, site, a, site_up)
                H += 1j * g_spinorbit * self.two_site_op(a, site, a_dag*sx, site_up)
        
        print("[BUILD] Adding mass term...")
        # Mass: energy cost for field occupation
        for site in range(self.n_sites):
            H += mass * self.single_site_op(n_field, site)
        
        self.H = H
        print(f"[BUILD] Hamiltonian complete. Dimension: {H.shape}")
        
        return H
    
    def init_spinor_vortex(self, x0, y0, winding_spatial=1, winding_spin=0):
        """
        Initialize with vortex structure in BOTH spatial and spin degrees of freedom.
        
        Args:
            winding_spatial: Spatial phase winding
            winding_spin: Spin texture winding (skyrmion-like)
        """
        print(f"[INIT] Creating spinor vortex:")
        print(f"  Spatial winding: {winding_spatial}")
        print(f"  Spin winding: {winding_spin}")
        
        local_states = []
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                dx = i - x0
                dy = j - y0
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                
                # Spatial phase
                phase_spatial = np.exp(1j * winding_spatial * theta)
                amp = np.exp(-r**2 / 2.0)
                
                # Spin texture (skyrmion-like if winding_spin != 0)
                # Spin points in direction that winds with angle
                if winding_spin != 0:
                    # Spin rotates as we go around vortex
                    spin_angle = winding_spin * theta
                    # State: cos(θ/2)|↑⟩ + e^(iφ)sin(θ/2)|↓⟩
                    # For simplicity: blend of up and down with phase
                    spin_up_amp = np.cos(spin_angle / 2)
                    spin_down_amp = np.exp(1j * spin_angle) * np.sin(spin_angle / 2)
                else:
                    # All spins up
                    spin_up_amp = 1.0
                    spin_down_amp = 0.0
                
                # Combined state: field occupation × spin
                # States: |0↑⟩=0, |0↓⟩=1, |1↑⟩=2, |1↓⟩=3
                
                # Occupied with up spin
                c_1up = amp * phase_spatial * spin_up_amp
                # Occupied with down spin  
                c_1down = amp * phase_spatial * spin_down_amp
                
                # State vector for this site
                psi_site = Qobj([
                    [1.0],           # |0↑⟩ - empty, spin up
                    [0.0],           # |0↓⟩ - empty, spin down
                    [c_1up],         # |1↑⟩ - occupied, spin up
                    [c_1down]        # |1↓⟩ - occupied, spin down
                ])
                
                psi_site = psi_site.unit()
                local_states.append(psi_site)
        
        self.state = tensor(local_states)
        print(f"[INIT] Spinor vortex state initialized")
        
        return self.state
    
    def evolve(self, t_max, dt):
        """Evolve and return final state."""
        if self.H is None or self.state is None:
            raise ValueError("Build Hamiltonian and initialize state first")
        
        times = np.arange(0, t_max + dt, dt)
        print(f"[EVOLVE] Evolving for t={t_max}, dt={dt}")
        
        result = mesolve(self.H, self.state, times, [], [])
        
        print(f"[EVOLVE] Complete")
        return result
    
    def measure_field_entropy(self, state, sites):
        """
        Measure entropy of field degrees of freedom at given sites.
        Traces out spin.
        """
        # This is tricky - need to trace out spin indices
        # For now, just measure total entropy of site subsystem
        rho = state.ptrace(sites)
        S = entropy_vn(rho)
        return S

def test_spinor_compression(winding_spatial=1, winding_spin=0, Nx=2, Ny=2,
                            t_max=2.0, dt=0.2):
    """
    Test if spinor vortices compress information.
    """
    print("\n" + "="*70)
    print("SPINOR COMPRESSION TEST")
    print(f"Spatial winding: {winding_spatial}")
    print(f"Spin winding: {winding_spin}")
    print("="*70)
    
    # Single pattern
    print("\n--- Single Spinor Pattern ---")
    lattice1 = SpinorLattice(Nx, Ny)
    lattice1.build_hamiltonian(J_hop=1.0, J_spin=0.5, g_spinorbit=0.2, mass=0.1)
    lattice1.init_spinor_vortex(Nx/2, Ny/2, winding_spatial, winding_spin)
    
    result1 = lattice1.evolve(t_max, dt)
    
    # Measure entropy (all sites)
    all_sites = list(range(lattice1.n_sites))
    S_single = lattice1.measure_field_entropy(result1.states[-1], all_sites)
    
    print(f"Single pattern entropy: S(one) = {S_single:.6f}")
    
    # Two patterns - need bigger lattice or different initialization
    # For now, just report single pattern result
    
    print("\n" + "="*70)
    print(f"Spinor pattern entropy: {S_single:.6f}")
    print("="*70)
    
    return {'S_single': float(S_single),
            'winding_spatial': winding_spatial,
            'winding_spin': winding_spin}

def main():
    parser = argparse.ArgumentParser(description="Test spinor lattice")
    parser.add_argument("--winding", type=int, default=1, help="Spatial winding")
    parser.add_argument("--spin_winding", type=int, default=0, help="Spin texture winding")
    parser.add_argument("--Nx", type=int, default=2)
    parser.add_argument("--Ny", type=int, default=2)
    parser.add_argument("--t_max", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="spinor_results.json")
    
    args = parser.parse_args()
    
    result = test_spinor_compression(
        args.winding, args.spin_winding,
        args.Nx, args.Ny, args.t_max, args.dt
    )
    
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {args.out}")

if __name__ == "__main__":
    main()