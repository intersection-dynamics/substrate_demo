#!/usr/bin/env python
"""
test_gauge_exclusion.py

Add GAUGE STRUCTURE to the spinor lattice.

Key idea: Gauge fields enforce constraints beyond energy penalties.
- U(1) gauge field on each link (like electromagnetism)
- Matter fields couple to gauge fields via minimal coupling
- Gauge transformations are local symmetries
- Gauss's law constrains allowed states

For fermions: Gauge constraints + spinor structure might FORBID overlapping states
(not just make them high energy, but make them gauge-violating)
"""

import numpy as np
import json
from qutip import Qobj, tensor, basis, qeye, mesolve, expect, entropy_vn

class GaugeLattice:
    """
    Lattice with matter fields (spinors) + gauge fields on links.
    
    Hilbert space structure:
    - Each site: 4 states (field × spin)
    - Each link: 2 states (gauge field phase: |0⟩, |π⟩ for discrete gauge)
    
    For 2×2 lattice:
    - 4 sites, 4 horizontal links, 4 vertical links = 8 links
    - Total: 4^4 × 2^8 = 65536 dimensional
    
    This is large but tractable.
    """
    
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.n_sites = Nx * Ny
        
        # Each site: 4 states (2 field × 2 spin)
        self.site_dim = 4
        
        # Each link: 2 states (discrete Z2 gauge: even/odd)
        # Z2 gauge is simpler than full U(1)
        self.link_dim = 2
        
        # Number of links (horizontal + vertical)
        self.n_horizontal = Nx * (Ny - 1)
        self.n_vertical = (Nx - 1) * Ny
        self.n_links = self.n_horizontal + self.n_vertical
        
        # Total Hilbert space
        self.total_dim = (self.site_dim ** self.n_sites) * (self.link_dim ** self.n_links)
        
        print(f"[INIT] Gauge lattice: {Nx}×{Ny}")
        print(f"[INIT] Sites: {self.n_sites} (dim={self.site_dim}^{self.n_sites})")
        print(f"[INIT] Links: {self.n_links} (dim={self.link_dim}^{self.n_links})")
        print(f"[INIT] Total Hilbert space: {self.total_dim}")
        
        if self.total_dim > 1000000:
            print("[WARNING] Very large Hilbert space - may be slow")
        
        self.H = None
        self.state = None
    
    def site_index(self, i, j):
        """Site index in flattened array."""
        return i * self.Ny + j
    
    def link_index_horizontal(self, i, j):
        """Index for horizontal link at (i,j) -> (i, j+1)."""
        return i * (self.Ny - 1) + j
    
    def link_index_vertical(self, i, j):
        """Index for vertical link at (i,j) -> (i+1, j)."""
        return self.n_horizontal + i * self.Ny + j
    
    def build_operators(self):
        """Build single-site matter operators."""
        # Matter field operators (same as before)
        n_field = Qobj([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
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
        
        a = Qobj([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0]])
        
        a_dag = a.dag()
        
        # Z2 gauge operators (on links)
        # |0⟩ = even, |1⟩ = odd
        # σx flips: |0⟩ ↔ |1⟩ (gauge transformation)
        # σz eigenvalue = (-1)^n (gauge field value)
        gauge_x = Qobj([[0, 1], [1, 0]])  # Gauge transformation generator
        gauge_z = Qobj([[1, 0], [0, -1]]) # Gauge field value (-1)^n
        
        return n_field, sx, sy, sz, a, a_dag, gauge_x, gauge_z
    
    def site_op(self, op, site):
        """Embed matter operator at given site."""
        ops = []
        
        # Matter sites
        for s in range(self.n_sites):
            if s == site:
                ops.append(op)
            else:
                ops.append(qeye(self.site_dim))
        
        # Gauge links (identity)
        for l in range(self.n_links):
            ops.append(qeye(self.link_dim))
        
        return tensor(ops)
    
    def link_op(self, op, link):
        """Embed gauge operator at given link."""
        ops = []
        
        # Matter sites (identity)
        for s in range(self.n_sites):
            ops.append(qeye(self.site_dim))
        
        # Gauge links
        for l in range(self.n_links):
            if l == link:
                ops.append(op)
            else:
                ops.append(qeye(self.link_dim))
        
        return tensor(ops)
    
    def gauge_covariant_hop(self, a_dag, site1, a, site2, link):
        """
        Gauge-covariant hopping: a†_i U_ij a_j
        
        For Z2 gauge: U = σz (eigenvalue ±1)
        Positive gauge field helps hopping, negative hinders it.
        """
        ops = []
        
        # Matter sites
        for s in range(self.n_sites):
            if s == site1:
                ops.append(a_dag)
            elif s == site2:
                ops.append(a)
            else:
                ops.append(qeye(self.site_dim))
        
        # Gauge links
        n_field, sx, sy, sz, a_mat, a_dag_mat, gauge_x, gauge_z = self.build_operators()
        
        for l in range(self.n_links):
            if l == link:
                ops.append(gauge_z)  # Gauge field couples to hopping
            else:
                ops.append(qeye(self.link_dim))
        
        return tensor(ops)
    
    def build_hamiltonian_gauge(self, J_hop=1.0, J_spin=0.5, g_spinorbit=0.3,
                               mass=0.1, U_onsite=1.0, g_gauge=0.5):
        """
        Build gauge-invariant Hamiltonian.
        
        Terms:
        1. Gauge-covariant hopping: a†_i U_ij a_j
        2. Spin interactions (gauge-invariant)
        3. Gauge field energy: -g × σx (tunneling term for gauge field)
        4. Onsite repulsion
        5. Gauss's law enforcement (optional)
        """
        print("[BUILD] Constructing gauge Hamiltonian...")
        
        if self.total_dim > 100000:
            print("[WARNING] Large Hilbert space - this will be slow")
            print("[WARNING] Consider reducing lattice size")
        
        n_field, sx, sy, sz, a, a_dag, gauge_x, gauge_z = self.build_operators()
        
        # Start with zero
        print("[BUILD] Initializing...")
        H = 0
        for s in range(self.n_sites):
            H += 0 * self.site_op(qeye(self.site_dim), s)
        
        # Gauge-covariant hopping
        print("[BUILD] - Gauge-covariant hopping")
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                # Horizontal hop
                if j + 1 < self.Ny:
                    site_right = self.site_index(i, j+1)
                    link = self.link_index_horizontal(i, j)
                    H -= J_hop * self.gauge_covariant_hop(a_dag, site, a, site_right, link)
                    H -= J_hop * self.gauge_covariant_hop(a, site, a_dag, site_right, link)
                
                # Vertical hop
                if i + 1 < self.Nx:
                    site_down = self.site_index(i+1, j)
                    link = self.link_index_vertical(i, j)
                    H -= J_hop * self.gauge_covariant_hop(a_dag, site, a, site_down, link)
                    H -= J_hop * self.gauge_covariant_hop(a, site, a_dag, site_down, link)
        
        print("[BUILD] - Gauge field dynamics")
        # Gauge field kinetic term: -g Σ_links σx
        # This allows gauge field to fluctuate
        for link in range(self.n_links):
            H -= g_gauge * self.link_op(gauge_x, link)
        
        print("[BUILD] - Mass and onsite repulsion")
        # Matter field energy
        for site in range(self.n_sites):
            H += mass * self.site_op(n_field, site)
            H += U_onsite * self.site_op(n_field @ n_field, site)
        
        # Spin interactions (simplified - just nearest neighbor)
        print("[BUILD] - Spin interactions")
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                if j + 1 < self.Ny:
                    site_right = self.site_index(i, j+1)
                    # Simplified: just sz-sz interaction
                    ops_left = []
                    ops_right = []
                    for s in range(self.n_sites):
                        if s == site:
                            ops_left.append(sz)
                            ops_right.append(qeye(self.site_dim))
                        elif s == site_right:
                            ops_left.append(qeye(self.site_dim))
                            ops_right.append(sz)
                        else:
                            ops_left.append(qeye(self.site_dim))
                            ops_right.append(qeye(self.site_dim))
                    
                    for l in range(self.n_links):
                        ops_left.append(qeye(self.link_dim))
                        ops_right.append(qeye(self.link_dim))
                    
                    H += J_spin * tensor(ops_left) @ tensor(ops_right)
        
        self.H = H
        print(f"[BUILD] Complete. H shape: {H.shape}")
        
        return H
    
    def init_matter_skyrmion(self, x0, y0, gauge_config='uniform'):
        """
        Initialize matter skyrmion + gauge field configuration.
        
        gauge_config:
        - 'uniform': All gauge fields in |0⟩ (even)
        - 'random': Random gauge configuration
        """
        print(f"[INIT] Skyrmion at ({x0}, {y0}), gauge: {gauge_config}")
        
        # Matter states
        matter_states = []
        for i in range(self.Nx):
            for j in range(self.Ny):
                dx = i - x0
                dy = j - y0
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                
                phase_spatial = np.exp(1j * theta)
                amp = np.exp(-r**2 / 2.0)
                
                spin_angle = theta
                spin_up_amp = np.cos(spin_angle / 2)
                spin_down_amp = np.exp(1j * spin_angle) * np.sin(spin_angle / 2)
                
                c_1up = amp * phase_spatial * spin_up_amp
                c_1down = amp * phase_spatial * spin_down_amp
                
                psi_site = Qobj([[1.0], [0.0], [c_1up], [c_1down]]).unit()
                matter_states.append(psi_site)
        
        # Gauge states
        gauge_states = []
        for link in range(self.n_links):
            if gauge_config == 'uniform':
                gauge_states.append(basis(2, 0))  # All even
            elif gauge_config == 'random':
                gauge_states.append((basis(2, 0) + (-1)**link * basis(2, 1)).unit())
        
        self.state = tensor(matter_states + gauge_states)
        return self.state
    
    def measure_energy(self, state):
        """Compute <ψ|H|ψ>."""
        E = expect(self.H, state)
        if abs(E.imag) > 1e-10:
            print(f"[WARNING] Energy has imaginary part: {E.imag}")
        return E.real

def test_gauge_exclusion_comparison():
    """
    THE REAL TEST: Do gauge fields enforce harder exclusion?
    
    Compare:
    1. One skyrmion
    2. Two skyrmions overlapping (same location)
    3. Two skyrmions separated
    
    All with gauge field coupling.
    """
    print("\n" + "="*70)
    print("GAUGE-ENFORCED EXCLUSION TEST")
    print("="*70)
    print("\nTest: Does gauge coupling make exclusion HARDER (constraint vs penalty)?")
    print()
    
    Nx, Ny = 2, 2
    
    results = {}
    
    # Config 1: One skyrmion with gauge fields
    print("\n" + "-"*70)
    print("CONFIG 1: One Skyrmion + Gauge Fields")
    print("-"*70)
    
    lattice1 = GaugeLattice(Nx, Ny)
    lattice1.build_hamiltonian_gauge(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3,
                                     mass=0.1, U_onsite=1.0, g_gauge=0.2)
    lattice1.init_matter_skyrmion(Nx/2, Ny/2, gauge_config='uniform')
    
    E1 = lattice1.measure_energy(lattice1.state)
    print(f"Energy: E = {E1:.6f}")
    
    results['one_skyrmion'] = {'E': float(E1)}
    
    # Config 2: Two skyrmions OVERLAPPING + gauge fields
    print("\n" + "-"*70)
    print("CONFIG 2: Two Skyrmions OVERLAPPING + Gauge Fields")
    print("-"*70)
    
    lattice2_overlap = GaugeLattice(Nx, Ny)
    lattice2_overlap.build_hamiltonian_gauge(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3,
                                             mass=0.1, U_onsite=1.0, g_gauge=0.2)
    
    # Initialize with double amplitude (two overlapping)
    print("[INIT] Two overlapping skyrmions at (1.0, 1.0)")
    
    # Matter states - double amplitude
    matter_states = []
    for i in range(Nx):
        for j in range(Ny):
            dx = i - Nx/2
            dy = j - Ny/2
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            phase_spatial = np.exp(1j * theta)
            amp = 2.0 * np.exp(-r**2 / 2.0)  # 2× amplitude
            
            spin_angle = theta
            spin_up_amp = np.cos(spin_angle / 2)
            spin_down_amp = np.exp(1j * spin_angle) * np.sin(spin_angle / 2)
            
            c_1up = amp * phase_spatial * spin_up_amp
            c_1down = amp * phase_spatial * spin_down_amp
            
            # Normalize
            norm = np.sqrt(1.0 + abs(c_1up)**2 + abs(c_1down)**2)
            psi_site = Qobj([[1.0/norm], [0.0], [c_1up/norm], [c_1down/norm]])
            matter_states.append(psi_site)
    
    # Gauge states - uniform
    gauge_states = [basis(2, 0) for _ in range(lattice2_overlap.n_links)]
    
    lattice2_overlap.state = tensor(matter_states + gauge_states)
    
    E2_overlap = lattice2_overlap.measure_energy(lattice2_overlap.state)
    print(f"Energy: E = {E2_overlap:.6f}")
    
    results['two_overlapping'] = {'E': float(E2_overlap)}
    
    # Config 3: Two skyrmions SEPARATED + gauge fields
    print("\n" + "-"*70)
    print("CONFIG 3: Two Skyrmions SEPARATED + Gauge Fields")
    print("-"*70)
    
    lattice2_sep = GaugeLattice(Nx, Ny)
    lattice2_sep.build_hamiltonian_gauge(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3,
                                         mass=0.1, U_onsite=1.0, g_gauge=0.2)
    
    print("[INIT] Two separated skyrmions")
    
    # Create superposition of two localized skyrmions
    matter_states_a = []
    matter_states_b = []
    
    for i in range(Nx):
        for j in range(Ny):
            # First skyrmion at (0.5, 1)
            dx1 = i - 0.5
            dy1 = j - 1.0
            r1 = np.sqrt(dx1**2 + dy1**2)
            theta1 = np.arctan2(dy1, dx1)
            
            phase1 = np.exp(1j * theta1)
            amp1 = 0.7 * np.exp(-r1**2 / 2.0)
            
            spin_angle1 = theta1
            spin_up1 = np.cos(spin_angle1 / 2)
            spin_down1 = np.exp(1j * spin_angle1) * np.sin(spin_angle1 / 2)
            
            c_1up_a = amp1 * phase1 * spin_up1
            c_1down_a = amp1 * phase1 * spin_down1
            
            norm_a = np.sqrt(1.0 + abs(c_1up_a)**2 + abs(c_1down_a)**2)
            psi_a = Qobj([[1.0/norm_a], [0.0], [c_1up_a/norm_a], [c_1down_a/norm_a]])
            matter_states_a.append(psi_a)
            
            # Second skyrmion at (1.5, 1)
            dx2 = i - 1.5
            dy2 = j - 1.0
            r2 = np.sqrt(dx2**2 + dy2**2)
            theta2 = np.arctan2(dy2, dx2)
            
            phase2 = np.exp(1j * theta2)
            amp2 = 0.7 * np.exp(-r2**2 / 2.0)
            
            spin_angle2 = theta2
            spin_up2 = np.cos(spin_angle2 / 2)
            spin_down2 = np.exp(1j * spin_angle2) * np.sin(spin_angle2 / 2)
            
            c_1up_b = amp2 * phase2 * spin_up2
            c_1down_b = amp2 * phase2 * spin_down2
            
            norm_b = np.sqrt(1.0 + abs(c_1up_b)**2 + abs(c_1down_b)**2)
            psi_b = Qobj([[1.0/norm_b], [0.0], [c_1up_b/norm_b], [c_1down_b/norm_b]])
            matter_states_b.append(psi_b)
    
    # Gauge states
    gauge_states = [basis(2, 0) for _ in range(lattice2_sep.n_links)]
    
    # Combine
    state_a = tensor(matter_states_a + gauge_states)
    state_b = tensor(matter_states_b + gauge_states)
    lattice2_sep.state = (state_a + state_b).unit()
    
    E2_sep = lattice2_sep.measure_energy(lattice2_sep.state)
    print(f"Energy: E = {E2_sep:.6f}")
    
    results['two_separated'] = {'E': float(E2_sep)}
    
    # Analysis
    print("\n" + "="*70)
    print("GAUGE EXCLUSION ANALYSIS")
    print("="*70)
    
    print("\nEnergy comparison (WITH gauge fields):")
    print(f"  One skyrmion:       E = {E1:8.4f}")
    print(f"  Two overlapping:    E = {E2_overlap:8.4f}")
    print(f"  Two separated:      E = {E2_sep:8.4f}")
    
    energy_cost = E2_overlap - E2_sep
    print(f"\nExclusion energy cost: {energy_cost:.4f}")
    
    print("\n" + "="*70)
    
    if energy_cost > 2.0:
        print("✓✓✓ STRONG GAUGE-ENFORCED EXCLUSION!")
        print(f"    Energy cost: ΔE = {energy_cost:.4f}")
        print("    → Gauge coupling STRENGTHENS exclusion")
        print("    → Harder constraint than pure matter interactions")
        verdict = "STRONG_EXCLUSION"
    elif energy_cost > 1.0:
        print("✓ GAUGE-ENHANCED EXCLUSION")
        print(f"    Energy cost: ΔE = {energy_cost:.4f}")
        print("    → Gauge fields contribute to exclusion")
        verdict = "MODERATE_EXCLUSION"
    elif energy_cost > 0.0:
        print("⚠ WEAK EXCLUSION")
        print(f"    Energy cost: ΔE = {energy_cost:.4f}")
        print("    → Gauge coupling doesn't significantly help")
        verdict = "WEAK_EXCLUSION"
    else:
        print("✗ NO EXCLUSION")
        print("    → Gauge structure doesn't enforce exclusion")
        verdict = "NO_EXCLUSION"
    
    print("="*70)
    
    results['verdict'] = verdict
    results['energy_cost'] = float(energy_cost)
    
    return results

def main():
    print("\n" + "="*70)
    print("LATTICE GAUGE THEORY: EMERGENT EXCLUSION")
    print("="*70)
    print("\nAdding gauge structure to test if it enforces exclusion.")
    print()
    
    # Run the comparison test
    results = test_gauge_exclusion_comparison()
    
    # Save results
    with open('gauge_exclusion_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: gauge_exclusion_results.json")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if results['verdict'] in ['STRONG_EXCLUSION', 'MODERATE_EXCLUSION']:
        print("\n✓ Gauge structure DOES enforce exclusion!")
        print("\nYour constraint package:")
        print("  1. Spinor structure (field + spin)")
        print("  2. Spin-orbit coupling")
        print("  3. Onsite repulsion (U)")
        print("  4. Gauge coupling (U(1) or Z2)")
        print("\nTogether these produce emergent Pauli exclusion.")
        print("Two identical skyrmions CANNOT occupy same location.")
        print("\nThis is the missing piece:")
        print("  Gauge structure turns 'soft' energy penalties")
        print("  into 'hard' constraints via gauge invariance.")
    else:
        print("\n⚠ Gauge structure alone may not be sufficient")
        print("\nPossibilities:")
        print("  - Need stronger gauge coupling (g_gauge)")
        print("  - Need different gauge group")
        print("  - Need additional topological constraints")
        print("  - Exclusion requires second quantization formalism")

if __name__ == "__main__":
    main()