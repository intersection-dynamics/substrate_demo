"""
Spin-Statistics Emergence Engine
================================
Many-body Hilbert space with explicit spin degrees of freedom.

Goal: See if antisymmetry emerges from spin structure without assuming it.

Key structures:
- Spatial degrees of freedom (position on lattice)
- Spin degrees of freedom (spin-1/2 per particle)
- Full exchange = exchange position AND spin
- Look for ground states that are antisymmetric under full exchange
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, kron, eye
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from itertools import product, permutations
from functools import reduce
import operator


# =============================================================================
# SPIN STRUCTURES
# =============================================================================

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_PLUS = (SIGMA_X + 1j * SIGMA_Y) / 2
SIGMA_MINUS = (SIGMA_X - 1j * SIGMA_Y) / 2
I2 = np.eye(2, dtype=complex)

# Spin states
SPIN_UP = np.array([1, 0], dtype=complex)    # |↑⟩ = |0⟩
SPIN_DOWN = np.array([0, 1], dtype=complex)  # |↓⟩ = |1⟩


def spin_dot(s1_ops: List[np.ndarray], s2_ops: List[np.ndarray]) -> np.ndarray:
    """
    Compute S₁·S₂ = (1/2)(S₊¹S₋² + S₋¹S₊²) + Sz¹Sz²
    
    For spin-1/2: S·S = (1/4)(σ·σ)
    """
    Sx1, Sy1, Sz1 = s1_ops
    Sx2, Sy2, Sz2 = s2_ops
    
    return (1/4) * (Sx1 @ Sx2 + Sy1 @ Sy2 + Sz1 @ Sz2)


def two_spin_singlet() -> np.ndarray:
    """
    Singlet state: |S=0, Sz=0⟩ = (|↑↓⟩ - |↓↑⟩)/√2
    
    ANTISYMMETRIC under spin exchange.
    """
    up_down = np.kron(SPIN_UP, SPIN_DOWN)
    down_up = np.kron(SPIN_DOWN, SPIN_UP)
    return (up_down - down_up) / np.sqrt(2)


def two_spin_triplet() -> List[np.ndarray]:
    """
    Triplet states: |S=1, Sz⟩
    
    SYMMETRIC under spin exchange.
    """
    # |1, +1⟩ = |↑↑⟩
    t_plus = np.kron(SPIN_UP, SPIN_UP)
    
    # |1, 0⟩ = (|↑↓⟩ + |↓↑⟩)/√2
    t_zero = (np.kron(SPIN_UP, SPIN_DOWN) + np.kron(SPIN_DOWN, SPIN_UP)) / np.sqrt(2)
    
    # |1, -1⟩ = |↓↓⟩
    t_minus = np.kron(SPIN_DOWN, SPIN_DOWN)
    
    return [t_plus, t_zero, t_minus]


# =============================================================================
# FULL HILBERT SPACE: POSITION ⊗ SPIN
# =============================================================================

class SpinfulParticleSpace:
    """
    Hilbert space for N particles with position AND spin.
    
    Basis states: |x₁, s₁; x₂, s₂; ...; xₙ, sₙ⟩
    where xᵢ ∈ {0, ..., L-1} and sᵢ ∈ {↑, ↓}
    
    Particles are initially DISTINGUISHABLE - we look for
    symmetry/antisymmetry to EMERGE.
    """
    
    def __init__(self, n_sites: int, n_particles: int):
        self.n_sites = n_sites
        self.n_particles = n_particles
        
        # Dimension: (n_sites * 2)^n_particles
        # Each particle has n_sites positions × 2 spin states
        self.single_particle_dim = n_sites * 2
        self.dim = self.single_particle_dim ** n_particles
        
        # Generate basis
        # State = tuple of (position, spin) for each particle
        # (x₁, s₁, x₂, s₂, ...) where s ∈ {0, 1} for ↑, ↓
        self.states = []
        self._generate_basis()
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
    def _generate_basis(self):
        """Generate all basis states."""
        single_particle_states = list(product(range(self.n_sites), range(2)))
        
        for config in product(single_particle_states, repeat=self.n_particles):
            # config = ((x₁, s₁), (x₂, s₂), ...)
            # Flatten to (x₁, s₁, x₂, s₂, ...)
            state = tuple(val for pair in config for val in pair)
            self.states.append(state)
    
    def get_position_spin(self, state: Tuple, particle: int) -> Tuple[int, int]:
        """Get (position, spin) for particle i from state tuple."""
        return state[2*particle], state[2*particle + 1]
    
    def positions(self, state: Tuple) -> Tuple[int, ...]:
        """Get all positions from state."""
        return tuple(state[2*i] for i in range(self.n_particles))
    
    def spins(self, state: Tuple) -> Tuple[int, ...]:
        """Get all spins from state."""
        return tuple(state[2*i + 1] for i in range(self.n_particles))
    
    def exchange_particles(self, state: Tuple, i: int, j: int) -> Tuple:
        """Exchange particles i and j (swap both position and spin)."""
        state_list = list(state)
        # Swap (xᵢ, sᵢ) with (xⱼ, sⱼ)
        state_list[2*i], state_list[2*j] = state_list[2*j], state_list[2*i]
        state_list[2*i+1], state_list[2*j+1] = state_list[2*j+1], state_list[2*i+1]
        return tuple(state_list)
    
    def has_spatial_double_occupation(self, state: Tuple) -> bool:
        """Check if any two particles are at the same position."""
        positions = self.positions(state)
        return len(positions) != len(set(positions))
    
    def occupation_at_site(self, state: Tuple, site: int) -> int:
        """Count particles at given site."""
        return sum(1 for i in range(self.n_particles) 
                   if state[2*i] == site)


# =============================================================================
# HAMILTONIAN WITH SPIN
# =============================================================================

def build_spinful_hamiltonian(space: SpinfulParticleSpace,
                               t: float = 1.0,
                               U: float = 0.0,
                               J_spin: float = 0.0,
                               J_exchange: float = 0.0) -> csr_matrix:
    """
    Build Hamiltonian for particles with spin.
    
    H = H_kinetic + H_onsite + H_spin + H_exchange
    
    Parameters
    ----------
    t : float
        Hopping amplitude (preserves spin)
    U : float  
        On-site repulsion when two particles at same site
    J_spin : float
        Heisenberg spin-spin coupling: J Σ_{⟨ij⟩} Sᵢ·Sⱼ
    J_exchange : float
        Direct exchange: couples to spin configuration when particles nearby
        
    Returns
    -------
    Sparse Hamiltonian matrix
    """
    dim = space.dim
    n_sites = space.n_sites
    n_particles = space.n_particles
    
    H = lil_matrix((dim, dim), dtype=complex)
    
    for idx, state in enumerate(space.states):
        # === Diagonal terms ===
        
        # On-site repulsion: U when two particles at same position
        if U != 0:
            positions = space.positions(state)
            for site in range(n_sites):
                n_at_site = positions.count(site)
                if n_at_site >= 2:
                    # n(n-1)/2 pairs at this site
                    H[idx, idx] += U * n_at_site * (n_at_site - 1) / 2
        
        # === Off-diagonal terms ===
        
        # Hopping: particle α hops, spin preserved
        for alpha in range(n_particles):
            x_alpha, s_alpha = space.get_position_spin(state, alpha)
            
            for dx in [-1, 1]:
                x_new = (x_alpha + dx) % n_sites
                
                # New state with particle α at x_new
                new_state = list(state)
                new_state[2*alpha] = x_new
                new_state = tuple(new_state)
                
                if new_state in space.state_to_idx:
                    new_idx = space.state_to_idx[new_state]
                    H[new_idx, idx] -= t
        
        # Spin-spin interaction between particles
        if J_spin != 0 and n_particles >= 2:
            for alpha in range(n_particles):
                for beta in range(alpha + 1, n_particles):
                    x_a, s_a = space.get_position_spin(state, alpha)
                    x_b, s_b = space.get_position_spin(state, beta)
                    
                    # Only interact if nearby (or same site)
                    dist = abs(x_a - x_b)
                    dist = min(dist, n_sites - dist)
                    
                    if dist <= 1:  # Nearest neighbors or same site
                        # Sᵅ·Sᵝ for spin-1/2:
                        # = (1/4) if parallel, -(3/4) if antiparallel (for singlet)
                        # More precisely, eigenvalue of S·S is:
                        # S(S+1) - 3/2 where S is total spin
                        # For S=1 (triplet): 2 - 3/2 = 1/2 → (1/4) per pair
                        # For S=0 (singlet): 0 - 3/2 = -3/2 → -(3/4) per pair
                        
                        # Diagonal part: Sz·Sz
                        sz_a = 0.5 if s_a == 0 else -0.5
                        sz_b = 0.5 if s_b == 0 else -0.5
                        H[idx, idx] += J_spin * sz_a * sz_b
                        
                        # Off-diagonal: (S+·S- + S-·S+)/2 = flip-flop
                        if s_a != s_b:
                            # Can flip both spins
                            new_state = list(state)
                            new_state[2*alpha + 1] = 1 - s_a
                            new_state[2*beta + 1] = 1 - s_b
                            new_state = tuple(new_state)
                            
                            if new_state in space.state_to_idx:
                                new_idx = space.state_to_idx[new_state]
                                H[new_idx, idx] += J_spin * 0.5
        
        # Direct exchange coupling
        # This explicitly couples spatial and spin exchange
        if J_exchange != 0 and n_particles >= 2:
            for alpha in range(n_particles):
                for beta in range(alpha + 1, n_particles):
                    x_a, s_a = space.get_position_spin(state, alpha)
                    x_b, s_b = space.get_position_spin(state, beta)
                    
                    dist = abs(x_a - x_b)
                    dist = min(dist, n_sites - dist)
                    
                    if dist == 1:  # Nearest neighbors
                        # Exchange term: P_αβ swaps both position and spin
                        exchanged = space.exchange_particles(state, alpha, beta)
                        if exchanged in space.state_to_idx:
                            ex_idx = space.state_to_idx[exchanged]
                            H[ex_idx, idx] += J_exchange
    
    return H.tocsr()


# =============================================================================
# SYMMETRY ANALYSIS
# =============================================================================

def compute_exchange_eigenvalue(psi: np.ndarray, space: SpinfulParticleSpace,
                                 i: int = 0, j: int = 1) -> complex:
    """
    Compute ⟨ψ|P_ij|ψ⟩ where P_ij exchanges particles i and j.
    
    For bosons: +1
    For fermions: -1
    """
    if space.n_particles < 2:
        return 1.0
    
    overlap = 0.0
    for idx, state in enumerate(space.states):
        exchanged = space.exchange_particles(state, i, j)
        if exchanged in space.state_to_idx:
            ex_idx = space.state_to_idx[exchanged]
            # ⟨ψ|P|ψ⟩ = Σ ψ*(state) ψ(P·state)
            overlap += np.conj(psi[idx]) * psi[ex_idx]
    
    return overlap


def analyze_symmetry_sectors(psi: np.ndarray, space: SpinfulParticleSpace) -> Dict[str, Any]:
    """
    Decompose wavefunction into symmetric and antisymmetric parts.
    
    For 2 particles:
    - Symmetric: (|ab⟩ + |ba⟩)/√2
    - Antisymmetric: (|ab⟩ - |ba⟩)/√2
    """
    if space.n_particles != 2:
        return {'error': 'Only implemented for 2 particles'}
    
    sym_weight = 0.0
    antisym_weight = 0.0
    
    for idx, state in enumerate(space.states):
        exchanged = space.exchange_particles(state, 0, 1)
        ex_idx = space.state_to_idx.get(exchanged)
        
        if ex_idx is not None and ex_idx > idx:
            # This is a pair (state, exchanged) with state < exchanged
            psi_ab = psi[idx]
            psi_ba = psi[ex_idx]
            
            # Symmetric component: (psi_ab + psi_ba)/√2
            sym = (psi_ab + psi_ba) / np.sqrt(2)
            # Antisymmetric component: (psi_ab - psi_ba)/√2
            antisym = (psi_ab - psi_ba) / np.sqrt(2)
            
            sym_weight += np.abs(sym)**2
            antisym_weight += np.abs(antisym)**2
        
        elif ex_idx == idx:
            # Self-exchange: state = exchanged (like |↑↑⟩ at same site)
            # This is purely symmetric
            sym_weight += np.abs(psi[idx])**2
    
    total = sym_weight + antisym_weight
    if total > 1e-10:
        sym_frac = sym_weight / total
        antisym_frac = antisym_weight / total
    else:
        sym_frac = 0.5
        antisym_frac = 0.5
    
    return {
        'symmetric_weight': sym_frac,
        'antisymmetric_weight': antisym_frac,
        'exchange_eigenvalue': compute_exchange_eigenvalue(psi, space),
        'is_fermionic': antisym_frac > 0.99,
        'is_bosonic': sym_frac > 0.99
    }


def analyze_spin_state(psi: np.ndarray, space: SpinfulParticleSpace) -> Dict[str, Any]:
    """
    Analyze the spin structure of the wavefunction.
    
    For 2 particles, decompose into singlet (S=0) and triplet (S=1) components.
    """
    if space.n_particles != 2:
        return {'error': 'Only implemented for 2 particles'}
    
    # Project onto spin states for each spatial configuration
    singlet_weight = 0.0
    triplet_weight = 0.0
    
    # Spin basis: (s₁, s₂) ∈ {(0,0), (0,1), (1,0), (1,1)}
    # Singlet: (|01⟩ - |10⟩)/√2 → coefficients: 0, 1/√2, -1/√2, 0
    # Triplet |1,+1⟩: |00⟩ → coefficients: 1, 0, 0, 0
    # Triplet |1,0⟩: (|01⟩ + |10⟩)/√2 → coefficients: 0, 1/√2, 1/√2, 0
    # Triplet |1,-1⟩: |11⟩ → coefficients: 0, 0, 0, 1
    
    singlet_coeffs = {(0, 1): 1/np.sqrt(2), (1, 0): -1/np.sqrt(2)}
    triplet_p1_coeffs = {(0, 0): 1.0}
    triplet_0_coeffs = {(0, 1): 1/np.sqrt(2), (1, 0): 1/np.sqrt(2)}
    triplet_m1_coeffs = {(1, 1): 1.0}
    
    # Group states by spatial configuration
    spatial_configs = {}
    for idx, state in enumerate(space.states):
        x1, s1, x2, s2 = state
        spatial = (x1, x2)
        spin = (s1, s2)
        
        if spatial not in spatial_configs:
            spatial_configs[spatial] = {}
        spatial_configs[spatial][spin] = psi[idx]
    
    # For each spatial config, compute singlet and triplet weights
    for spatial, spin_amplitudes in spatial_configs.items():
        # Singlet projection
        singlet_amp = sum(singlet_coeffs.get(spin, 0) * amp 
                         for spin, amp in spin_amplitudes.items())
        singlet_weight += np.abs(singlet_amp)**2
        
        # Triplet projections
        for triplet_coeffs in [triplet_p1_coeffs, triplet_0_coeffs, triplet_m1_coeffs]:
            triplet_amp = sum(triplet_coeffs.get(spin, 0) * amp
                             for spin, amp in spin_amplitudes.items())
            triplet_weight += np.abs(triplet_amp)**2
    
    total = singlet_weight + triplet_weight
    if total > 1e-10:
        singlet_frac = singlet_weight / total
        triplet_frac = triplet_weight / total
    else:
        singlet_frac = 0.0
        triplet_frac = 0.0
    
    return {
        'singlet_weight': singlet_frac,
        'triplet_weight': triplet_frac,
        'is_singlet': singlet_frac > 0.99,
        'is_triplet': triplet_frac > 0.99
    }


def measure_spatial_exclusion(psi: np.ndarray, space: SpinfulParticleSpace) -> float:
    """
    Measure probability of spatial exclusion (no two particles at same site).
    
    True fermions: 1.0 (Pauli exclusion in space for same spin)
    """
    exclusion_prob = 0.0
    for idx, state in enumerate(space.states):
        if not space.has_spatial_double_occupation(state):
            exclusion_prob += np.abs(psi[idx])**2
    return exclusion_prob


def full_analysis(psi: np.ndarray, space: SpinfulParticleSpace) -> Dict[str, Any]:
    """Complete analysis of wavefunction."""
    results = {}
    
    # Exchange symmetry
    sym = analyze_symmetry_sectors(psi, space)
    results.update(sym)
    
    # Spin structure  
    spin = analyze_spin_state(psi, space)
    results['singlet_weight'] = spin['singlet_weight']
    results['triplet_weight'] = spin['triplet_weight']
    
    # Spatial exclusion
    results['spatial_exclusion'] = measure_spatial_exclusion(psi, space)
    
    # Character determination
    if results['antisymmetric_weight'] > 0.99:
        results['character'] = 'FERMIONIC'
    elif results['symmetric_weight'] > 0.99:
        results['character'] = 'BOSONIC'
    else:
        results['character'] = 'MIXED'
    
    return results


# =============================================================================
# EXPERIMENTS
# =============================================================================

def find_ground_states(H: csr_matrix, k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Find k lowest eigenstates."""
    dim = H.shape[0]
    if dim <= 20:
        H_dense = H.toarray()
        energies, states = np.linalg.eigh(H_dense)
        return energies[:k], states[:, :k]
    else:
        k = min(k, dim - 2)
        energies, states = eigsh(H, k=k, which='SA')
        idx = np.argsort(energies)
        return energies[idx], states[:, idx]


def scan_spin_coupling(n_sites: int, t: float, U: float,
                       J_values: List[float]) -> Dict[str, Any]:
    """
    Scan over spin coupling strength J_spin.
    
    Question: Does the ground state become antisymmetric (fermionic)
    for some value of J?
    """
    space = SpinfulParticleSpace(n_sites, n_particles=2)
    
    results = {
        'J_values': J_values,
        'antisym_weight': [],
        'singlet_weight': [],
        'spatial_exclusion': [],
        'exchange_eigenvalue': [],
        'ground_energy': []
    }
    
    for J in J_values:
        H = build_spinful_hamiltonian(space, t=t, U=U, J_spin=J)
        E, psi = find_ground_states(H, k=1)
        psi = psi[:, 0]
        
        analysis = full_analysis(psi, space)
        
        results['antisym_weight'].append(analysis['antisymmetric_weight'])
        results['singlet_weight'].append(analysis['singlet_weight'])
        results['spatial_exclusion'].append(analysis['spatial_exclusion'])
        results['exchange_eigenvalue'].append(np.real(analysis['exchange_eigenvalue']))
        results['ground_energy'].append(E[0])
    
    return results


def scan_exchange_coupling(n_sites: int, t: float, U: float,
                           J_ex_values: List[float]) -> Dict[str, Any]:
    """
    Scan over direct exchange coupling J_exchange.
    
    This directly couples to the exchange operator P_ij.
    """
    space = SpinfulParticleSpace(n_sites, n_particles=2)
    
    results = {
        'J_ex_values': J_ex_values,
        'antisym_weight': [],
        'singlet_weight': [],
        'spatial_exclusion': [],
        'exchange_eigenvalue': [],
        'ground_energy': []
    }
    
    for J_ex in J_ex_values:
        H = build_spinful_hamiltonian(space, t=t, U=U, J_exchange=J_ex)
        E, psi = find_ground_states(H, k=1)
        psi = psi[:, 0]
        
        analysis = full_analysis(psi, space)
        
        results['antisym_weight'].append(analysis['antisymmetric_weight'])
        results['singlet_weight'].append(analysis['singlet_weight'])
        results['spatial_exclusion'].append(analysis['spatial_exclusion'])
        results['exchange_eigenvalue'].append(np.real(analysis['exchange_eigenvalue']))
        results['ground_energy'].append(E[0])
    
    return results


def compare_spectrum(n_sites: int, t: float, U: float, J_spin: float = 0.0) -> Dict[str, Any]:
    """
    Analyze the full low-energy spectrum.
    
    Classify each eigenstate by symmetry character.
    """
    space = SpinfulParticleSpace(n_sites, n_particles=2)
    H = build_spinful_hamiltonian(space, t=t, U=U, J_spin=J_spin)
    
    n_states = min(20, space.dim)
    energies, states = find_ground_states(H, k=n_states)
    
    results = {
        'energies': energies.tolist(),
        'states': []
    }
    
    for i in range(n_states):
        psi = states[:, i]
        analysis = full_analysis(psi, space)
        analysis['energy'] = energies[i]
        analysis['state_index'] = i
        results['states'].append(analysis)
    
    return results


def run_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point.
    
    Modes:
    - 'scan_J_spin': scan over Heisenberg coupling
    - 'scan_J_exchange': scan over direct exchange
    - 'spectrum': analyze full spectrum
    - 'single': single parameter point
    """
    mode = params.get('mode', 'single')
    n_sites = params.get('n_sites', 4)
    t = params.get('t', 1.0)
    U = params.get('U', 0.0)
    J_spin = params.get('J_spin', 0.0)
    J_exchange = params.get('J_exchange', 0.0)
    
    if mode == 'scan_J_spin':
        J_values = params.get('J_values', [-2, -1, -0.5, 0, 0.5, 1, 2])
        return scan_spin_coupling(n_sites, t, U, J_values)
    
    elif mode == 'scan_J_exchange':
        J_ex_values = params.get('J_ex_values', [-2, -1, -0.5, 0, 0.5, 1, 2])
        return scan_exchange_coupling(n_sites, t, U, J_ex_values)
    
    elif mode == 'spectrum':
        return compare_spectrum(n_sites, t, U, J_spin)
    
    else:  # single
        space = SpinfulParticleSpace(n_sites, n_particles=2)
        H = build_spinful_hamiltonian(space, t=t, U=U, J_spin=J_spin, J_exchange=J_exchange)
        E, psi = find_ground_states(H, k=1)
        
        return {
            'params': params,
            'ground_energy': E[0],
            **full_analysis(psi[:, 0], space)
        }