#!/usr/bin/env python
"""
quantum_substrate_evolution.py

Time evolution of quantum lattice with entanglement tracking.
Tests whether substrate dynamics naturally produce patterns with
monogamous entanglement structure.

Implements:
1. Local + non-local Hamiltonians (nearest-neighbor + defrag-like)
2. Unitary time evolution
3. Entanglement tracking over time
4. Multiple pattern evolution (test exclusion)
"""

import numpy as np
import argparse
import os
from pathlib import Path
import json

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

try:
    from qutip import Qobj, tensor, basis, qeye, sigmax, sigmay, sigmaz, destroy, mesolve
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    print("[WARNING] QuTiP not available")


class QuantumSubstrateEvolver:
    """
    Evolves quantum lattice system and tracks entanglement structure.
    """
    
    def __init__(self, Nx, Ny, local_dim=2):
        self.Nx = Nx
        self.Ny = Ny
        self.n_sites = Nx * Ny
        self.local_dim = local_dim
        
        if not QUTIP_AVAILABLE:
            raise ImportError("QuTiP required for time evolution")
        
        # Build Hamiltonian
        self.H = None
        self.state = None
        
        print(f"[INIT] Quantum substrate evolver: {Nx}x{Ny}, local_dim={local_dim}")
    
    def site_index(self, i, j):
        """2D to 1D index."""
        return i * self.Ny + j
    
    def build_hamiltonian(self, J_nn=1.0, J_defrag=0.1, mass=0.5):
        """
        Build Hamiltonian with:
        1. Nearest-neighbor hopping (kinetic-like term)
        2. Defrag interaction (long-range, density-dependent)
        3. Mass term
        
        For qubits (local_dim=2):
        H = -J_nn Σ_<ij> (σ_i^+ σ_j^- + h.c.) + m Σ_i σ_i^z
            + J_defrag Σ_{ij} (1/r_ij) n_i n_j
        
        where n_i = (1 + σ_i^z)/2 is number operator
        """
        print("[BUILD] Constructing Hamiltonian...")
        
        if self.local_dim != 2:
            raise NotImplementedError("Currently only implemented for qubits (local_dim=2)")
        
        # Build identity and Pauli operators for all sites
        I_list = [qeye(2) for _ in range(self.n_sites)]
        sx_list = [sigmax() for _ in range(self.n_sites)]
        sz_list = [sigmaz() for _ in range(self.n_sites)]
        
        def single_site_op(op, site):
            """Insert operator at given site, identity elsewhere."""
            ops = [qeye(2) if i != site else op for i in range(self.n_sites)]
            return tensor(ops)
        
        def two_site_op(op1, site1, op2, site2):
            """Two-site operator."""
            ops = [qeye(2) for _ in range(self.n_sites)]
            ops[site1] = op1
            ops[site2] = op2
            return tensor(ops)
        
        # Start with zero Hamiltonian
        H = 0.0 * tensor(I_list)
        
        # 1. Nearest-neighbor hopping
        print("[BUILD] Adding nearest-neighbor terms...")
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                # Neighbor sites (with periodic BC)
                neighbors = [
                    self.site_index((i+1) % self.Nx, j),  # right
                    self.site_index(i, (j+1) % self.Ny),  # up
                ]
                
                for nb in neighbors:
                    # Hopping: a†_i a_j + h.c. (approximated with σ+σ- for qubits)
                    sp = (sigmax() + 1j * sigmay()) / 2  # σ+ = raising operator
                    sm = (sigmax() - 1j * sigmay()) / 2  # σ- = lowering operator
                    
                    H -= J_nn * two_site_op(sp, site, sm, nb)
                    H -= J_nn * two_site_op(sm, site, sp, nb)
        
        # 2. Mass term (on-site energy)
        print("[BUILD] Adding mass term...")
        for site in range(self.n_sites):
            H += mass * single_site_op(sigmaz(), site)
        
        # 3. Defrag interaction (simplified - only nearby sites for tractability)
        print("[BUILD] Adding defrag interaction...")
        max_range = 2  # Only interact within distance 2 to keep H manageable
        for site1 in range(self.n_sites):
            i1, j1 = divmod(site1, self.Ny)
            for site2 in range(site1 + 1, self.n_sites):
                i2, j2 = divmod(site2, self.Ny)
                
                dx = min(abs(i2 - i1), self.Nx - abs(i2 - i1))  # periodic
                dy = min(abs(j2 - j1), self.Ny - abs(j2 - j1))
                r = np.sqrt(dx**2 + dy**2)
                
                if r < max_range and r > 0:
                    # Number operators: n = (1 + σz)/2
                    n1_op = (single_site_op(qeye(2), site1) + single_site_op(sigmaz(), site1)) / 2
                    n2_op = (single_site_op(qeye(2), site2) + single_site_op(sigmaz(), site2)) / 2
                    
                    # Interaction strength decays with distance
                    strength = J_defrag / (r**2 + 0.1)
                    
                    # This is expensive - just add density-density interaction
                    H += strength * single_site_op(sigmaz(), site1) * single_site_op(sigmaz(), site2)
        
        self.H = H
        print(f"[BUILD] Hamiltonian constructed. Dimension: {H.shape}")
        
        return H
    
    def init_state(self, pattern_type='single_vortex', **kwargs):
        """Initialize quantum state with specific pattern."""
        if pattern_type == 'single_vortex':
            x0 = kwargs.get('x0', self.Nx / 2)
            y0 = kwargs.get('y0', self.Ny / 2)
            w = kwargs.get('winding', 1)
            
            # Create product state with vortex phase structure
            local_states = []
            for i in range(self.Nx):
                for j in range(self.Ny):
                    dx = i - x0
                    dy = j - y0
                    r = np.sqrt(dx**2 + dy**2)
                    theta = np.arctan2(dy, dx)
                    
                    phase = np.exp(1j * w * theta)
                    amp = np.exp(-r**2 / 4.0)
                    
                    # Qubit state: |0⟩ + amp*phase |1⟩
                    psi_local = (basis(2, 0) + amp * phase * basis(2, 1)).unit()
                    local_states.append(psi_local)
            
            self.state = tensor(local_states)
        
        elif pattern_type == 'two_vortex':
            x1 = kwargs.get('x1', self.Nx / 3)
            y1 = kwargs.get('y1', self.Ny / 2)
            x2 = kwargs.get('x2', 2 * self.Nx / 3)
            y2 = kwargs.get('y2', self.Ny / 2)
            w = kwargs.get('winding', 1)
            
            # Two vortices - create superposition
            local_states = []
            for i in range(self.Nx):
                for j in range(self.Ny):
                    dx1, dy1 = i - x1, j - y1
                    dx2, dy2 = i - x2, j - y2
                    r1 = np.sqrt(dx1**2 + dy1**2)
                    r2 = np.sqrt(dx2**2 + dy2**2)
                    theta1 = np.arctan2(dy1, dx1)
                    theta2 = np.arctan2(dy2, dx2)
                    
                    phase1 = np.exp(1j * w * theta1)
                    phase2 = np.exp(1j * w * theta2)
                    amp1 = np.exp(-r1**2 / 4.0)
                    amp2 = np.exp(-r2**2 / 4.0)
                    
                    # Superpose both contributions
                    coeff = amp1 * phase1 + amp2 * phase2
                    psi_local = (basis(2, 0) + coeff * basis(2, 1)).unit()
                    local_states.append(psi_local)
            
            self.state = tensor(local_states)
        
        return self.state
    
    def evolve(self, t_max, dt, measurement_times=None):
        """
        Evolve system unitarily and track entanglement.
        
        Args:
            t_max: Total evolution time
            dt: Time step
            measurement_times: Times at which to measure entanglement
        
        Returns:
            Dictionary with entanglement data over time
        """
        if self.H is None:
            raise ValueError("Hamiltonian not built. Call build_hamiltonian() first.")
        if self.state is None:
            raise ValueError("State not initialized. Call init_state() first.")
        
        if measurement_times is None:
            measurement_times = np.arange(0, t_max + dt, dt)
        
        print(f"[EVOLVE] Running evolution: t_max={t_max}, dt={dt}")
        print(f"[EVOLVE] Measurements at {len(measurement_times)} times")
        
        # QuTiP's mesolve for unitary evolution
        result = mesolve(self.H, self.state, measurement_times, [], [])
        
        print(f"[EVOLVE] Evolution complete")
        
        return result
    
    def measure_entanglement_at_state(self, state, pattern_sites, env_sites):
        """
        Measure entanglement between pattern and environment for given state.
        """
        # Compute reduced density matrices
        rho_pattern = state.ptrace(pattern_sites)
        
        # von Neumann entropy
        S_pattern = -sum([(x * np.log(x)).real if x > 1e-15 else 0 
                          for x in rho_pattern.eigenenergies()])
        
        return S_pattern
    
    def track_entanglement_evolution(self, result, pattern_sites, env_sites):
        """
        Track entanglement entropy over time from evolution result.
        """
        times = result.times
        entropies = []
        
        print("[TRACK] Computing entanglement over time...")
        for i, state in enumerate(result.states):
            S = self.measure_entanglement_at_state(state, pattern_sites, env_sites)
            entropies.append(S)
            if i % max(1, len(result.states) // 10) == 0:
                print(f"  t={times[i]:.3f}: S={S:.6f}")
        
        return times, entropies


def main():
    parser = argparse.ArgumentParser(description="Quantum substrate time evolution")
    parser.add_argument("--Nx", type=int, default=3, help="Lattice size (keep small!)")
    parser.add_argument("--Ny", type=int, default=3)
    parser.add_argument("--J_nn", type=float, default=1.0, help="Nearest-neighbor coupling")
    parser.add_argument("--J_defrag", type=float, default=0.5, help="Defrag coupling")
    parser.add_argument("--mass", type=float, default=0.1, help="Mass parameter")
    parser.add_argument("--t_max", type=float, default=5.0, help="Evolution time")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step for output")
    parser.add_argument("--pattern", type=str, default='single_vortex', 
                       choices=['single_vortex', 'two_vortex'])
    parser.add_argument("--winding", type=int, default=1)
    parser.add_argument("--out", type=str, default="quantum_evolution_output")
    
    args = parser.parse_args()
    
    print("\n=== Quantum Substrate Evolution ===\n")
    
    # Initialize
    evolver = QuantumSubstrateEvolver(args.Nx, args.Ny)
    
    # Build Hamiltonian
    evolver.build_hamiltonian(J_nn=args.J_nn, J_defrag=args.J_defrag, mass=args.mass)
    
    # Initialize state
    print(f"\n[INIT] Creating {args.pattern} state (w={args.winding})")
    if args.pattern == 'single_vortex':
        evolver.init_state('single_vortex', x0=args.Nx/2, y0=args.Ny/2, winding=args.winding)
        pattern_sites = [evolver.site_index(i, j) 
                        for i in range(args.Nx) for j in range(args.Ny)
                        if (i - args.Nx/2)**2 + (j - args.Ny/2)**2 < 2.0]
    else:  # two_vortex
        evolver.init_state('two_vortex', winding=args.winding)
        # Pattern sites are central region
        pattern_sites = [evolver.site_index(i, j) 
                        for i in range(args.Nx) for j in range(args.Ny)
                        if 1 <= i <= args.Nx-2]
    
    env_sites = [s for s in range(evolver.n_sites) if s not in pattern_sites]
    
    print(f"[INIT] Pattern sites: {pattern_sites}")
    print(f"[INIT] Environment sites: {env_sites}")
    
    # Evolve
    measurement_times = np.arange(0, args.t_max + args.dt, args.dt)
    result = evolver.evolve(args.t_max, args.dt, measurement_times)
    
    # Track entanglement
    times, entropies = evolver.track_entanglement_evolution(result, pattern_sites, env_sites)
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    
    results_dict = {
        'times': list(times),  # Already a list from QuTiP, but ensure it's JSON serializable
        'entropies': entropies,
        'pattern_sites': pattern_sites,
        'env_sites': env_sites,
        'parameters': {
            'Nx': args.Nx,
            'Ny': args.Ny,
            'J_nn': args.J_nn,
            'J_defrag': args.J_defrag,
            'mass': args.mass,
            'winding': args.winding,
            'pattern_type': args.pattern
        }
    }
    
    out_file = os.path.join(args.out, 'entanglement_evolution.json')
    with open(out_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n[SAVED] Results → {out_file}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Initial entropy: S(0) = {entropies[0]:.6f}")
    print(f"Final entropy:   S({args.t_max}) = {entropies[-1]:.6f}")
    print(f"Change: ΔS = {entropies[-1] - entropies[0]:.6f}")
    
    print("\n=== Evolution Complete ===\n")


if __name__ == "__main__":
    main()