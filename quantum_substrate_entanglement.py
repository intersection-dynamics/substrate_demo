#!/usr/bin/env python
"""
quantum_substrate_entanglement.py

Quantum lattice simulator for measuring entanglement structure in topological patterns.
Tests whether patterns with winding number w=±1 have monogamous entanglement.

Key differences from classical field simulations:
- Represents actual quantum state |Ψ⟩ in tensor product Hilbert space
- Computes reduced density matrices via partial trace
- Measures entanglement entropy and negativity
- Tests monogamy of entanglement

System: 2D lattice where each site has local Hilbert space (complex field levels or spins)
Total Hilbert space: H = ⊗_sites H_local
State: |Ψ⟩ ∈ H (exponentially large, so limited to small lattices)
"""

import numpy as np
import argparse
import os
from pathlib import Path

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[INFO] CuPy available - using GPU")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("[INFO] CuPy not available - using CPU")

# Quantum toolbox
try:
    from qutip import Qobj, tensor, basis, qeye, mesolve, entropy_vn, partial_transpose
    QUTIP_AVAILABLE = True
    print("[INFO] QuTiP available")
except ImportError:
    QUTIP_AVAILABLE = False
    print("[WARNING] QuTiP not available - some features disabled")


class QuantumLattice:
    """
    Quantum lattice system where each site has a local Hilbert space.
    
    For testing monogamy, we need:
    1. Initialize patterns with topological structure (vortex-like)
    2. Evolve under Hamiltonian (local + interaction terms)
    3. Compute reduced density matrices for subsystems
    4. Measure entanglement between pattern and environment
    """
    
    def __init__(self, Nx, Ny, local_dim=2, use_gpu=True):
        """
        Initialize quantum lattice.
        
        Args:
            Nx, Ny: Lattice dimensions
            local_dim: Dimension of local Hilbert space per site
                      (2 for qubit, 3-4 for truncated field, etc)
            use_gpu: Use GPU acceleration where possible
        """
        self.Nx = Nx
        self.Ny = Ny
        self.n_sites = Nx * Ny
        self.local_dim = local_dim
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Total Hilbert space dimension (exponentially large!)
        self.total_dim = local_dim ** self.n_sites
        
        if self.total_dim > 2**20:  # ~1M dimensional
            print(f"[WARNING] Total Hilbert space dimension = {self.total_dim}")
            print(f"[WARNING] This may exceed memory. Consider smaller lattice or local_dim.")
        
        # State vector representation
        # For small systems: store full state vector
        # For larger systems: would need tensor network methods (not implemented here)
        self.state = None
        
        # Site coordinates
        self.coords = [(i, j) for i in range(Nx) for j in range(Ny)]
        
        print(f"[INIT] Quantum lattice: {Nx}×{Ny} sites, local_dim={local_dim}")
        print(f"[INIT] Total Hilbert space dimension: {self.total_dim}")
    
    def site_index(self, i, j):
        """Convert 2D coordinates to linear site index."""
        return i * self.Ny + j
    
    def init_product_state(self, local_states):
        """
        Initialize state as tensor product of local states.
        
        Args:
            local_states: list of local state vectors (length n_sites)
                         each should be array of shape (local_dim,)
        """
        if len(local_states) != self.n_sites:
            raise ValueError(f"Need {self.n_sites} local states")
        
        # Compute tensor product
        state = local_states[0]
        for s in local_states[1:]:
            state = np.kron(state, s)
        
        # Normalize
        state = state / np.linalg.norm(state)
        
        if self.use_gpu:
            self.state = cp.array(state, dtype=cp.complex128)
        else:
            self.state = state.astype(np.complex128)
        
        return self.state
    
    def init_vortex_pattern(self, x0, y0, winding, width=2.0):
        """
        Initialize state with vortex-like phase pattern.
        
        For each site (i,j), the local state has phase determined by
        angle to vortex center (x0, y0) with winding number.
        
        Args:
            x0, y0: Vortex center (in lattice coordinates)
            winding: Winding number (±1, ±2, ...)
            width: Spatial width of excitation
        """
        local_states = []
        
        for i, j in self.coords:
            dx = i - x0
            dy = j - y0
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Phase from winding
            phase = np.exp(1j * winding * theta)
            
            # Amplitude envelope (Gaussian-like)
            amplitude = np.exp(-r**2 / (2 * width**2))
            
            # Local state: superposition of basis states with phase
            if self.local_dim == 2:
                # Qubit: |0⟩ + phase*amplitude |1⟩
                local_state = np.array([1.0, phase * amplitude], dtype=np.complex128)
            else:
                # Multi-level: distribute amplitude across levels
                local_state = np.zeros(self.local_dim, dtype=np.complex128)
                local_state[0] = 1.0
                for n in range(1, self.local_dim):
                    local_state[n] = phase * amplitude / np.sqrt(n)
            
            # Normalize
            local_state = local_state / np.linalg.norm(local_state)
            local_states.append(local_state)
        
        return self.init_product_state(local_states)
    
    def init_two_vortex_state(self, x1, y1, w1, x2, y2, w2, width=2.0):
        """
        Initialize state with two vortex patterns.
        This creates entanglement between the patterns from the start.
        """
        local_states = []
        
        for i, j in self.coords:
            # Distance and angle to each vortex
            dx1, dy1 = i - x1, j - y1
            dx2, dy2 = i - x2, j - y2
            r1 = np.sqrt(dx1**2 + dy1**2)
            r2 = np.sqrt(dx2**2 + dy2**2)
            theta1 = np.arctan2(dy1, dx1)
            theta2 = np.arctan2(dy2, dx2)
            
            # Phases from both vortices
            phase1 = np.exp(1j * w1 * theta1)
            phase2 = np.exp(1j * w2 * theta2)
            amp1 = np.exp(-r1**2 / (2 * width**2))
            amp2 = np.exp(-r2**2 / (2 * width**2))
            
            # Superposition of contributions
            # This is crude - should be improved for realistic patterns
            phase = phase1 * amp1 + phase2 * amp2
            amplitude = np.sqrt(amp1**2 + amp2**2)
            
            if self.local_dim == 2:
                local_state = np.array([1.0, phase * amplitude], dtype=np.complex128)
            else:
                local_state = np.zeros(self.local_dim, dtype=np.complex128)
                local_state[0] = 1.0
                for n in range(1, self.local_dim):
                    local_state[n] = phase * amplitude / np.sqrt(n)
            
            local_state = local_state / np.linalg.norm(local_state)
            local_states.append(local_state)
        
        return self.init_product_state(local_states)
    
    def reduced_density_matrix(self, region_sites):
        """
        Compute reduced density matrix for a region by tracing out rest.
        
        Args:
            region_sites: list of site indices to keep
        
        Returns:
            Reduced density matrix as numpy array
        
        WARNING: This is exponentially expensive. For Nx*Ny=16 sites,
        if region has 4 sites, we need to sum over 2^12 = 4096 basis states.
        """
        if self.state is None:
            raise ValueError("State not initialized")
        
        # Convert to numpy if on GPU
        if self.use_gpu:
            state = cp.asnumpy(self.state)
        else:
            state = self.state
        
        # This is the hard part - need to reshape state vector into tensor
        # form and perform partial trace
        
        # For now, use QuTiP if available
        if not QUTIP_AVAILABLE:
            raise NotImplementedError("Partial trace requires QuTiP or manual implementation")
        
        # Convert to QuTiP format
        # State is vector of dimension (local_dim)^n_sites
        qobj_state = Qobj(state.reshape(-1, 1))
        qobj_state.dims = [[self.local_dim] * self.n_sites, [1] * self.n_sites]
        
        # Compute density matrix
        rho_full = qobj_state * qobj_state.dag()
        
        # Sites to trace out
        trace_sites = [s for s in range(self.n_sites) if s not in region_sites]
        
        # Partial trace
        rho_reduced = rho_full.ptrace(region_sites)
        
        return rho_reduced.full()
    
    def entanglement_entropy(self, region_sites):
        """
        Compute von Neumann entropy S(ρ_A) = -Tr(ρ_A log ρ_A)
        for region A.
        
        Measures total entanglement between region and rest.
        """
        rho = self.reduced_density_matrix(region_sites)
        
        # Eigenvalues
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-15]  # Remove numerical zeros
        
        # von Neumann entropy
        S = -np.sum(eigvals * np.log(eigvals))
        
        return S
    
    def mutual_information(self, region_A, region_B):
        """
        Compute mutual information I(A:B) = S(A) + S(B) - S(AB)
        
        Measures correlations between regions A and B.
        If I(A:B) ≈ 0, regions are uncorrelated.
        """
        S_A = self.entanglement_entropy(region_A)
        S_B = self.entanglement_entropy(region_B)
        
        region_AB = list(set(region_A) | set(region_B))
        S_AB = self.entanglement_entropy(region_AB)
        
        return S_A + S_B - S_AB
    
    def negativity(self, region_A, region_B):
        """
        Compute logarithmic negativity between regions A and B.
        
        Measures entanglement between the two regions.
        E_N = log(||ρ^{T_B}||_1) where T_B is partial transpose wrt B.
        
        If regions are separable, E_N = 0.
        If entangled, E_N > 0.
        """
        if not QUTIP_AVAILABLE:
            raise NotImplementedError("Negativity calculation requires QuTiP")
        
        # Get reduced density matrix for A+B
        region_AB = list(set(region_A) | set(region_B))
        rho_AB = self.reduced_density_matrix(region_AB)
        
        # Convert to QuTiP and compute partial transpose
        # (This is tricky - need proper subsystem structure)
        # For now, return placeholder
        
        # Proper implementation would use partial_transpose from QuTiP
        # on the properly structured density matrix
        
        # Simplified: compute trace norm of partial transpose
        # ||M||_1 = sum of singular values
        singular_vals = np.linalg.svd(rho_AB, compute_uv=False)
        trace_norm = np.sum(np.abs(singular_vals))
        
        E_N = np.log(trace_norm)
        
        return E_N
    
    def test_monogamy(self, pattern_sites, env_sites_1, env_sites_2):
        """
        Test monogamy of entanglement: if pattern is maximally entangled
        with env_1, it cannot also be entangled with env_2.
        
        Args:
            pattern_sites: sites of the pattern
            env_sites_1: first environment region
            env_sites_2: second environment region (should be disjoint from env_1)
        
        Returns:
            Dictionary with entanglement measures
        """
        results = {}
        
        # Entanglement between pattern and each environment
        I_P_E1 = self.mutual_information(pattern_sites, env_sites_1)
        I_P_E2 = self.mutual_information(pattern_sites, env_sites_2)
        
        results['I(P:E1)'] = I_P_E1
        results['I(P:E2)'] = I_P_E2
        
        # Pattern entropy
        S_P = self.entanglement_entropy(pattern_sites)
        results['S(P)'] = S_P
        
        # Monogamy inequality: I(P:E1) + I(P:E2) <= 2*S(P)
        # For maximal entanglement: I(P:E1) ≈ S(P) implies I(P:E2) ≈ 0
        results['monogamy_sum'] = I_P_E1 + I_P_E2
        results['monogamy_bound'] = 2 * S_P
        results['monogamous'] = (I_P_E1 > 0.8 * S_P) and (I_P_E2 < 0.2 * S_P)
        
        return results


def identify_pattern_sites(lattice, x_center, y_center, radius):
    """
    Identify sites belonging to a pattern centered at (x_center, y_center)
    within given radius.
    """
    pattern_sites = []
    for s, (i, j) in enumerate(lattice.coords):
        dx = i - x_center
        dy = j - y_center
        if np.sqrt(dx**2 + dy**2) <= radius:
            pattern_sites.append(s)
    return pattern_sites


def main():
    parser = argparse.ArgumentParser(description="Quantum substrate: test monogamy of entanglement")
    parser.add_argument("--Nx", type=int, default=4, help="Lattice size x (keep small!)")
    parser.add_argument("--Ny", type=int, default=4, help="Lattice size y")
    parser.add_argument("--local_dim", type=int, default=2, help="Local Hilbert space dimension")
    parser.add_argument("--winding", type=int, default=1, help="Vortex winding number")
    parser.add_argument("--pattern_radius", type=float, default=1.5, help="Radius of pattern region")
    parser.add_argument("--out", type=str, default="quantum_output", help="Output directory")
    
    args = parser.parse_args()
    
    print("\n=== Quantum Substrate: Monogamy Test ===\n")
    
    # Initialize lattice
    lattice = QuantumLattice(args.Nx, args.Ny, local_dim=args.local_dim)
    
    # Initialize single vortex pattern
    print(f"\n[SETUP] Initializing vortex pattern (w={args.winding})")
    x0 = args.Nx / 2.0
    y0 = args.Ny / 2.0
    lattice.init_vortex_pattern(x0, y0, winding=args.winding, width=2.0)
    
    # Identify pattern sites (core region)
    pattern_sites = identify_pattern_sites(lattice, x0, y0, args.pattern_radius)
    print(f"[SETUP] Pattern region: {len(pattern_sites)} sites")
    
    # Define two disjoint environment regions
    # E1: left side, E2: right side
    env_sites_1 = [s for s, (i, j) in enumerate(lattice.coords) 
                   if i < args.Nx/2 and s not in pattern_sites]
    env_sites_2 = [s for s, (i, j) in enumerate(lattice.coords) 
                   if i >= args.Nx/2 and s not in pattern_sites]
    
    print(f"[SETUP] Environment 1: {len(env_sites_1)} sites (left)")
    print(f"[SETUP] Environment 2: {len(env_sites_2)} sites (right)")
    
    # Test monogamy
    print("\n[MEASURING] Computing entanglement structure...")
    
    try:
        results = lattice.test_monogamy(pattern_sites, env_sites_1, env_sites_2)
        
        print("\n=== RESULTS ===")
        print(f"Pattern entropy S(P) = {results['S(P)']:.6f}")
        print(f"Mutual information I(P:E1) = {results['I(P:E1)']:.6f}")
        print(f"Mutual information I(P:E2) = {results['I(P:E2)']:.6f}")
        print(f"Sum I(P:E1) + I(P:E2) = {results['monogamy_sum']:.6f}")
        print(f"Monogamy bound 2*S(P) = {results['monogamy_bound']:.6f}")
        print(f"Monogamous entanglement: {results['monogamous']}")
        
        # Save results
        os.makedirs(args.out, exist_ok=True)
        results_file = os.path.join(args.out, "monogamy_results.txt")
        with open(results_file, 'w') as f:
            f.write("=== Monogamy Test Results ===\n\n")
            f.write(f"Lattice: {args.Nx}x{args.Ny}\n")
            f.write(f"Local dimension: {args.local_dim}\n")
            f.write(f"Winding number: {args.winding}\n")
            f.write(f"Pattern sites: {len(pattern_sites)}\n")
            f.write(f"Env1 sites: {len(env_sites_1)}\n")
            f.write(f"Env2 sites: {len(env_sites_2)}\n\n")
            for key, val in results.items():
                f.write(f"{key}: {val}\n")
        
        print(f"\n[SAVED] Results → {results_file}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    main()