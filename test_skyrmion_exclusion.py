#!/usr/bin/env python
"""
test_skyrmion_exclusion.py  (with U(1) lattice gauge field)

THE REAL TEST: Can two skyrmions occupy the same location?

For fermions: The substrate should FORBID this
- Energy diverges
- State is dynamically unstable
- System naturally pushes them apart

For bosons: The substrate should ALLOW this
- Energy is reasonable
- State is stable
- Patterns can pile up (condensate-like)

This tests Pauli exclusion at the substrate level - not imposed, but emergent.

NEW: We introduce a background U(1) gauge field via Peierls phases on the
hopping and spin-orbit terms. The parameter `gauge_flux` controls the
flux per plaquette (in units of 2π) in a Landau-style gauge.
"""

import numpy as np
import json
from qutip import Qobj, tensor, qeye, mesolve, expect
import matplotlib.pyplot as plt


class SpinorLatticeExclusion:
    """
    Spinor lattice for testing exclusion principle, with optional U(1) gauge field.
    """
    
    def __init__(self, Nx, Ny, gauge_flux=0.0):
        """
        Nx, Ny      : lattice size
        gauge_flux  : flux per plaquette (in units of 2π), implemented
                      via a Peierls phase in Landau gauge:
                      
                        horizontal hops: phase = 1
                        vertical hops:   phase = exp( i * 2π * gauge_flux * x )
                      
                      Set gauge_flux = 0.0 to disable the gauge field.
        """
        self.Nx = Nx
        self.Ny = Ny
        self.n_sites = Nx * Ny
        self.site_dim = 4  # 2 field × 2 spin
        self.total_dim = self.site_dim ** self.n_sites
        
        self.gauge_flux = gauge_flux  # store gauge parameter
        
        print(f"[INIT] Spinor lattice: {Nx}×{Ny} = {self.n_sites} sites")
        print(f"[INIT] Hilbert space dimension: {self.total_dim}")
        print(f"[INIT] Gauge flux per plaquette (in units of 2π): {self.gauge_flux}")
        
        self.H = None
        self.state = None
    
    def site_index(self, i, j):
        return i * self.Ny + j
    
    def build_operators(self):
        """Build single-site operators."""
        # Field occupation: 0 for 'vacuum' subspace, 1 for 'excited' subspace
        n_field = Qobj([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        # Spin operators (act in both vacuum/excited subspaces the same way)
        sx = Qobj([[0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]]) / 2
        
        sy = Qobj([[0, -1j, 0, 0],
                   [1j, 0, 0, 0],
                   [0, 0, 0, -1j],
                   [0, 0, 1j, 0]]) / 2
        
        sz = Qobj([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]]) / 2
        
        # Field ladder operators
        a = Qobj([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0]])
        
        a_dag = a.dag()
        
        return n_field, sx, sy, sz, a, a_dag
    
    def single_site_op(self, op, site):
        ops = [qeye(self.site_dim) for _ in range(self.n_sites)]
        ops[site] = op
        return tensor(ops)
    
    def two_site_op(self, op1, site1, op2, site2):
        ops = [qeye(self.site_dim) for _ in range(self.n_sites)]
        ops[site1] = op1
        ops[site2] = op2
        return tensor(ops)
    
    def _link_phase(self, i, j, nb_i, nb_j):
        """
        U(1) gauge link phase between site (i,j) and (nb_i, nb_j).
        
        We use a simple Landau gauge:
          A_x = 0
          A_y = 2π * gauge_flux * x
          
        So:
          horizontal hops (x-direction): phase = 1
          vertical hops (y-direction):   phase = exp(i * 2π * gauge_flux * i)
        
        Returns complex phase factor U_ij such that hopping term is:
          a_i† U_ij a_j + h.c.
        """
        if self.gauge_flux == 0.0:
            return 1.0 + 0j
        
        if nb_i == i + 1 and nb_j == j:
            # horizontal hop to the right
            return 1.0 + 0j
        elif nb_i == i and nb_j == j + 1:
            # vertical hop "up"
            phase = np.exp(1j * 2.0 * np.pi * self.gauge_flux * i)
            return phase
        else:
            # Shouldn't happen with our nearest-neighbor pattern
            return 1.0 + 0j
    
    def build_hamiltonian(self, J_hop=1.0, J_spin=0.5, g_spinorbit=0.3, 
                          mass=0.1, U_onsite=2.0):
        """
        Build Hamiltonian with ONSITE REPULSION and optional U(1) gauge coupling.
        
        U_onsite: Cost for having high field density at one site.
                  (Here implemented as n^2; with our 0/1 'n', it's effectively
                   a stiffening term rather than true n(n-1) double occupancy.)
        """
        print("[BUILD] Constructing Hamiltonian...")
        
        n_field, sx, sy, sz, a, a_dag = self.build_operators()
        
        H = 0 * tensor([qeye(self.site_dim) for _ in range(self.n_sites)])
        
        # Hopping with Peierls phases
        print("[BUILD] - Hopping terms (with gauge field if gauge_flux ≠ 0)")
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                neighbors = []
                if i + 1 < self.Nx:
                    neighbors.append(self.site_index(i+1, j))
                if j + 1 < self.Ny:
                    neighbors.append(self.site_index(i, j+1))
                
                for nb in neighbors:
                    nb_i = nb // self.Ny
                    nb_j = nb % self.Ny
                    
                    U_ij = self._link_phase(i, j, nb_i, nb_j)
                    # a_i† U_ij a_j + h.c. with correct complex conjugate
                    H -= J_hop * (U_ij * self.two_site_op(a_dag, site, a, nb)
                                  + np.conj(U_ij) * self.two_site_op(a, site, a_dag, nb))
        
        # Spin-spin (no gauge coupling directly on spin part here)
        print("[BUILD] - Spin interactions")
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                neighbors = []
                if i + 1 < self.Nx:
                    neighbors.append(self.site_index(i+1, j))
                if j + 1 < self.Ny:
                    neighbors.append(self.site_index(i, j+1))
                
                for nb in neighbors:
                    H += J_spin * self.two_site_op(sx, site, sx, nb)
                    H += J_spin * self.two_site_op(sy, site, sy, nb)
                    H += J_spin * self.two_site_op(sz, site, sz, nb)
        
        # Spin-orbit with gauge phases on the field hop
        print("[BUILD] - Spin-orbit coupling (gauged)")
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                if i + 1 < self.Nx:
                    site_right = self.site_index(i+1, j)
                    U_ij = self._link_phase(i, j, i+1, j)
                    hop_right = self.two_site_op(a_dag @ sz, site, a, site_right)
                    hop_right = U_ij * hop_right
                    # Add both H and H† to ensure Hermiticity
                    H -= 1j * g_spinorbit * hop_right
                    H += 1j * g_spinorbit * hop_right.dag()
                
                if j + 1 < self.Ny:
                    site_up = self.site_index(i, j+1)
                    U_ij = self._link_phase(i, j, i, j+1)
                    hop_up = self.two_site_op(a_dag @ sx, site, a, site_up)
                    hop_up = U_ij * hop_up
                    H -= 1j * g_spinorbit * hop_up
                    H += 1j * g_spinorbit * hop_up.dag()
        
        # Mass
        print("[BUILD] - Mass term")
        for site in range(self.n_sites):
            H += mass * self.single_site_op(n_field, site)
        
        # ONSITE REPULSION - key for exclusion!
        print(f"[BUILD] - Onsite repulsion (U={U_onsite})")
        for site in range(self.n_sites):
            # Using n^2 as a stiffness term for high local excitation density
            n = self.single_site_op(n_field, site)
            H += U_onsite * n @ n
        
        self.H = H
        print(f"[BUILD] Complete. H shape: {H.shape}")
        
        return H
    
    def init_skyrmion(self, x0, y0, amplitude=1.0):
        """Initialize single skyrmion with given amplitude."""
        local_states = []
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                dx = i - x0
                dy = j - y0
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                
                phase_spatial = np.exp(1j * theta)
                amp = amplitude * np.exp(-r**2 / 2.0)
                
                spin_angle = theta
                spin_up_amp = np.cos(spin_angle / 2)
                spin_down_amp = np.exp(1j * spin_angle) * np.sin(spin_angle / 2)
                
                c_1up = amp * phase_spatial * spin_up_amp
                c_1down = amp * phase_spatial * spin_down_amp
                
                psi_site = Qobj([[1.0],
                                 [0.0],
                                 [c_1up],
                                 [c_1down]]).unit()
                local_states.append(psi_site)
        
        self.state = tensor(local_states)
        return self.state
    
    def init_two_skyrmions_overlapping(self, x0, y0):
        """
        Initialize TWO skyrmions at THE SAME LOCATION.
        
        This is the forbidden state for fermions.
        For bosons, this would be like a condensate.
        """
        local_states = []
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                dx = i - x0
                dy = j - y0
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                
                # Double the amplitude - trying to put TWO skyrmions here
                phase_spatial = np.exp(1j * theta)
                amp = 2.0 * np.exp(-r**2 / 2.0)  # 2× amplitude
                
                spin_angle = theta
                spin_up_amp = np.cos(spin_angle / 2)
                spin_down_amp = np.exp(1j * spin_angle) * np.sin(spin_angle / 2)
                
                c_1up = amp * phase_spatial * spin_up_amp
                c_1down = amp * phase_spatial * spin_down_amp
                
                # Renormalize
                norm = np.sqrt(1.0 + abs(c_1up)**2 + abs(c_1down)**2)
                
                psi_site = Qobj([[1.0/norm],
                                 [0.0],
                                 [c_1up/norm],
                                 [c_1down/norm]])
                local_states.append(psi_site)
        
        self.state = tensor(local_states)
        return self.state
    
    def measure_energy(self, state):
        """Compute <ψ|H|ψ>."""
        E = expect(self.H, state)
        if abs(E.imag) > 1e-10:
            print(f"[WARNING] Energy has imaginary part: {E.imag}")
        return E.real
    
    def measure_total_occupation(self, state):
        """Total field occupation across all sites."""
        n_field, _, _, _, _, _ = self.build_operators()
        
        N_total = 0
        for site in range(self.n_sites):
            N_total += expect(self.single_site_op(n_field, site), state)
        
        return N_total
    
    def evolve(self, t_max, dt):
        """Evolve state."""
        times = np.arange(0, t_max + dt, dt)
        print(f"[EVOLVE] t_max={t_max}, dt={dt}")
        result = mesolve(self.H, self.state, times, [], [])
        print("[EVOLVE] Complete")
        return result


def test_exclusion_principle(Nx=2, Ny=2, U_onsite=2.0, gauge_flux=0.0):
    """
    THE TEST: What happens when we try to put two skyrmions at same location?
    Includes the effect of a background U(1) gauge field if gauge_flux ≠ 0.
    """
    print("\n" + "="*70)
    print("SKYRMION EXCLUSION TEST")
    print("="*70)
    print("\nTest: Can two identical skyrmions occupy the same location?")
    print("  Fermions: NO - high energy cost, dynamically forbidden")
    print("  Bosons: YES - low energy, can condense")
    print(f"\nGauge flux per plaquette (in units of 2π): {gauge_flux}")
    print()
    
    results = {}
    
    # Configuration 1: Empty (baseline)
    print("\n" + "-"*70)
    print("CONFIG 1: Empty State (baseline)")
    print("-"*70)
    
    lattice0 = SpinorLatticeExclusion(Nx, Ny, gauge_flux=gauge_flux)
    lattice0.build_hamiltonian(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3, 
                               mass=0.1, U_onsite=U_onsite)
    
    # Start in empty state (all sites in |0↑⟩)
    empty_states = [Qobj([[1.0], [0.0], [0.0], [0.0]]) for _ in range(Nx*Ny)]
    state0 = tensor(empty_states)
    
    E0 = lattice0.measure_energy(state0)
    N0 = 0.0
    
    print(f"Energy: E = {E0:.6f}")
    print(f"Occupation: N = {N0:.6f}")
    
    results['empty'] = {'E': float(E0), 'N': float(N0)}
    
    # Configuration 2: One skyrmion
    print("\n" + "-"*70)
    print("CONFIG 2: One Skyrmion")
    print("-"*70)
    
    lattice1 = SpinorLatticeExclusion(Nx, Ny, gauge_flux=gauge_flux)
    lattice1.build_hamiltonian(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3, 
                               mass=0.1, U_onsite=U_onsite)
    lattice1.init_skyrmion(Nx/2, Ny/2, amplitude=1.0)
    
    E1 = lattice1.measure_energy(lattice1.state)
    N1 = lattice1.measure_total_occupation(lattice1.state)
    
    print(f"Energy: E = {E1:.6f}")
    print(f"Occupation: N = {N1:.6f}")
    print(f"Energy per particle: E/N = {E1/N1:.6f}")
    
    results['one_skyrmion'] = {'E': float(E1), 'N': float(N1), 'E_per_N': float(E1/N1)}
    
    # Configuration 3: Two skyrmions OVERLAPPING (same location)
    print("\n" + "-"*70)
    print("CONFIG 3: Two Skyrmions OVERLAPPING (same location)")
    print("-"*70)
    
    lattice2_overlap = SpinorLatticeExclusion(Nx, Ny, gauge_flux=gauge_flux)
    lattice2_overlap.build_hamiltonian(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3,
                                       mass=0.1, U_onsite=U_onsite)
    lattice2_overlap.init_two_skyrmions_overlapping(Nx/2, Ny/2)
    
    E2_overlap = lattice2_overlap.measure_energy(lattice2_overlap.state)
    N2_overlap = lattice2_overlap.measure_total_occupation(lattice2_overlap.state)
    
    print(f"Energy: E = {E2_overlap:.6f}")
    print(f"Occupation: N = {N2_overlap:.6f}")
    print(f"Energy per particle: E/N = {E2_overlap/N2_overlap:.6f}")
    
    results['two_overlapping'] = {'E': float(E2_overlap), 'N': float(N2_overlap), 
                                  'E_per_N': float(E2_overlap/N2_overlap)}
    
    # Configuration 4: Two skyrmions SEPARATED
    print("\n" + "-"*70)
    print("CONFIG 4: Two Skyrmions SEPARATED")
    print("-"*70)
    
    lattice2_sep = SpinorLatticeExclusion(Nx, Ny, gauge_flux=gauge_flux)
    lattice2_sep.build_hamiltonian(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3,
                                   mass=0.1, U_onsite=U_onsite)
    
    # Create superposition of two separated skyrmions
    # This is still an approximate "two-pattern" state.
    lattice2_sep.init_skyrmion(0.5, Ny/2, amplitude=0.7)
    state_a = lattice2_sep.state
    
    lattice2_sep.init_skyrmion(Nx-0.5, Ny/2, amplitude=0.7)
    state_b = lattice2_sep.state
    
    lattice2_sep.state = (state_a + state_b).unit()
    
    E2_sep = lattice2_sep.measure_energy(lattice2_sep.state)
    N2_sep = lattice2_sep.measure_total_occupation(lattice2_sep.state)
    
    print(f"Energy: E = {E2_sep:.6f}")
    print(f"Occupation: N = {N2_sep:.6f}")
    print(f"Energy per particle: E/N = {E2_sep/N2_sep:.6f}")
    
    results['two_separated'] = {'E': float(E2_sep), 'N': float(N2_sep),
                                'E_per_N': float(E2_sep/N2_sep)}
    
    # Analysis
    print("\n" + "="*70)
    print("EXCLUSION ANALYSIS")
    print("="*70)
    
    print("\nEnergy comparison:")
    print(f"  Empty:              E = {E0:8.4f}")
    print(f"  One skyrmion:       E = {E1:8.4f}  (E/N = {E1/N1:.4f})")
    print(f"  Two overlapping:    E = {E2_overlap:8.4f}  (E/N = {E2_overlap/N2_overlap:.4f})")
    print(f"  Two separated:      E = {E2_sep:8.4f}  (E/N = {E2_sep/N2_sep:.4f})")
    
    # Key test: Is overlapping more expensive than separated?
    E_overlap_excess = E2_overlap - 2*E1
    E_sep_excess = E2_sep - 2*E1
    
    print(f"\nExcess energy (compared to 2× single):")
    print(f"  Overlapping: ΔE = {E_overlap_excess:.4f}")
    print(f"  Separated:   ΔE = {E_sep_excess:.4f}")
    
    # Direct comparison: overlapping vs separated
    energy_cost = E2_overlap - E2_sep
    print(f"\nDirect comparison:")
    print(f"  Overlapping - Separated = {energy_cost:.4f}")
    
    print("\n" + "="*70)
    
    if energy_cost >
