#!/usr/bin/env python
"""
test_two_skyrmion_compression.py

THE KEY TEST: Do two skyrmions compress their information (bosonic)
or maintain independent information (fermionic)?

This is the direct test of your hypothesis:
- Bosons: carry copyable information → S(two) < 2×S(one)
- Fermions: carry unique information → S(two) ≈ 2×S(one)
"""

import numpy as np
import json
from qutip import Qobj, tensor, basis, qeye, mesolve, entropy_vn

class SpinorLatticeForTwoPatterns:
    """
    Spinor lattice optimized for two-pattern test.
    Uses 3×2 lattice = 6 sites = 4^6 = 4096 dimensional (tractable).
    """
    
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.n_sites = Nx * Ny
        self.site_dim = 4  # 2 field × 2 spin
        self.total_dim = self.site_dim ** self.n_sites
        
        print(f"[INIT] Spinor lattice: {Nx}×{Ny} = {self.n_sites} sites")
        print(f"[INIT] Hilbert space dimension: {self.site_dim}^{self.n_sites} = {self.total_dim}")
        
        if self.total_dim > 100000:
            print(f"[WARNING] Large Hilbert space - evolution may be slow")
        
        self.H = None
        self.state = None
    
    def site_index(self, i, j):
        return i * self.Ny + j
    
    def build_operators(self):
        """Build single-site operators (same as before)."""
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
    
    def build_hamiltonian(self, J_hop=1.0, J_spin=0.5, g_spinorbit=0.3, mass=0.1):
        """Build spinor Hamiltonian."""
        print("[BUILD] Constructing Hamiltonian...")
        
        n_field, sx, sy, sz, a, a_dag = self.build_operators()
        
        H = 0 * tensor([qeye(self.site_dim) for _ in range(self.n_sites)])
        
        # Hopping
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                neighbors = []
                if i + 1 < self.Nx:
                    neighbors.append(self.site_index(i+1, j))
                if j + 1 < self.Ny:
                    neighbors.append(self.site_index(i, j+1))
                
                for nb in neighbors:
                    H -= J_hop * self.two_site_op(a_dag, site, a, nb)
                    H -= J_hop * self.two_site_op(a, site, a_dag, nb)
        
        # Spin-spin
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
        
        # Spin-orbit
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)
                
                if i + 1 < self.Nx:
                    site_right = self.site_index(i+1, j)
                    H -= 1j * g_spinorbit * self.two_site_op(a_dag @ sz, site, a, site_right)
                    H += 1j * g_spinorbit * self.two_site_op(a, site, a_dag @ sz, site_right)
                
                if j + 1 < self.Ny:
                    site_up = self.site_index(i, j+1)
                    H -= 1j * g_spinorbit * self.two_site_op(a_dag @ sx, site, a, site_up)
                    H += 1j * g_spinorbit * self.two_site_op(a, site, a_dag @ sx, site_up)
        
        # Mass
        for site in range(self.n_sites):
            H += mass * self.single_site_op(n_field, site)
        
        self.H = H
        print(f"[BUILD] Complete. H shape: {H.shape}")
        
        return H
    
    def init_one_skyrmion(self, x0, y0):
        """Initialize with single skyrmion."""
        print(f"[INIT] Single skyrmion at ({x0:.1f}, {y0:.1f})")
        
        local_states = []
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                dx = i - x0
                dy = j - y0
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                
                # Skyrmion: w_spatial=1, w_spin=1
                phase_spatial = np.exp(1j * theta)
                amp = np.exp(-r**2 / 2.0)
                
                spin_angle = theta
                spin_up_amp = np.cos(spin_angle / 2)
                spin_down_amp = np.exp(1j * spin_angle) * np.sin(spin_angle / 2)
                
                c_1up = amp * phase_spatial * spin_up_amp
                c_1down = amp * phase_spatial * spin_down_amp
                
                psi_site = Qobj([[1.0], [0.0], [c_1up], [c_1down]]).unit()
                local_states.append(psi_site)
        
        self.state = tensor(local_states)
        return self.state
    
    def init_two_skyrmions(self, x1, y1, x2, y2):
        """Initialize with TWO skyrmions."""
        print(f"[INIT] Two skyrmions:")
        print(f"  Skyrmion 1 at ({x1:.1f}, {y1:.1f})")
        print(f"  Skyrmion 2 at ({x2:.1f}, {y2:.1f})")
        
        local_states = []
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                # Distance to each skyrmion
                dx1, dy1 = i - x1, j - y1
                dx2, dy2 = i - x2, j - y2
                r1 = np.sqrt(dx1**2 + dy1**2)
                r2 = np.sqrt(dx2**2 + dy2**2)
                theta1 = np.arctan2(dy1, dx1)
                theta2 = np.arctan2(dy2, dx2)
                
                # Contribution from skyrmion 1
                phase1 = np.exp(1j * theta1)
                amp1 = np.exp(-r1**2 / 2.0)
                spin_angle1 = theta1
                spin1_up = np.cos(spin_angle1 / 2)
                spin1_down = np.exp(1j * spin_angle1) * np.sin(spin_angle1 / 2)
                
                # Contribution from skyrmion 2
                phase2 = np.exp(1j * theta2)
                amp2 = np.exp(-r2**2 / 2.0)
                spin_angle2 = theta2
                spin2_up = np.cos(spin_angle2 / 2)
                spin2_down = np.exp(1j * spin_angle2) * np.sin(spin_angle2 / 2)
                
                # Superpose contributions (this is the key!)
                # If they compress, this superposition collapses
                # If they don't compress, maintains separate structure
                total_amp = amp1 + amp2
                if total_amp > 0:
                    # Weighted average of spin states
                    avg_up = (amp1 * phase1 * spin1_up + amp2 * phase2 * spin2_up) / total_amp
                    avg_down = (amp1 * phase1 * spin1_down + amp2 * phase2 * spin2_down) / total_amp
                else:
                    avg_up = 1.0
                    avg_down = 0.0
                
                c_1up = total_amp * avg_up
                c_1down = total_amp * avg_down
                
                psi_site = Qobj([[1.0], [0.0], [c_1up], [c_1down]]).unit()
                local_states.append(psi_site)
        
        self.state = tensor(local_states)
        return self.state
    
    def evolve(self, t_max, dt):
        """Evolve state."""
        times = np.arange(0, t_max + dt, dt)
        print(f"[EVOLVE] t_max={t_max}, dt={dt}")
        result = mesolve(self.H, self.state, times, [], [])
        print("[EVOLVE] Complete")
        return result

def measure_bipartite_entropy(state, n_sites):
    """Measure entanglement between left and right halves."""
    left_sites = list(range(n_sites // 2))
    right_sites = list(range(n_sites // 2, n_sites))
    
    rho_left = state.ptrace(left_sites)
    S = entropy_vn(rho_left)
    
    return S

def test_skyrmion_compression(Nx=3, Ny=2, t_max=4.0, dt=0.25):
    """
    THE TEST: Do two skyrmions compress their information?
    """
    print("\n" + "="*70)
    print("SKYRMION COMPRESSION TEST")
    print("="*70)
    print("\nDirect test of fermion vs boson hypothesis:")
    print("  Bosons: S(two) < 2×S(one) - information compresses")
    print("  Fermions: S(two) ≈ 2×S(one) - information doesn't compress")
    print()
    
    # Test 1: Single skyrmion
    print("\n" + "-"*70)
    print("TEST 1: Single Skyrmion")
    print("-"*70)
    
    lattice1 = SpinorLatticeForTwoPatterns(Nx, Ny)
    lattice1.build_hamiltonian(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3, mass=0.1)
    lattice1.init_one_skyrmion(Nx/2, Ny/2)
    
    result1 = lattice1.evolve(t_max, dt)
    S_one = measure_bipartite_entropy(result1.states[-1], lattice1.n_sites)
    
    print(f"\nSingle skyrmion entropy: S(one) = {S_one:.6f}")
    
    # Test 2: Two skyrmions
    print("\n" + "-"*70)
    print("TEST 2: Two Skyrmions")
    print("-"*70)
    
    lattice2 = SpinorLatticeForTwoPatterns(Nx, Ny)
    lattice2.build_hamiltonian(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3, mass=0.1)
    
    # Place two skyrmions with small separation
    sep = 1.0
    lattice2.init_two_skyrmions(Nx/2 - sep/2, Ny/2, Nx/2 + sep/2, Ny/2)
    
    result2 = lattice2.evolve(t_max, dt)
    S_two = measure_bipartite_entropy(result2.states[-1], lattice2.n_sites)
    
    print(f"\nTwo skyrmions entropy: S(two) = {S_two:.6f}")
    
    # Analysis
    print("\n" + "="*70)
    print("COMPRESSION ANALYSIS")
    print("="*70)
    
    S_expected = 2 * S_one
    compression_ratio = S_two / S_expected
    
    print(f"\nEntropy measurements:")
    print(f"  Single skyrmion:           S(one) = {S_one:.6f}")
    print(f"  Two skyrmions:             S(two) = {S_two:.6f}")
    print(f"  Expected if independent: 2×S(one) = {S_expected:.6f}")
    
    print(f"\nCompression ratio: S(two) / 2×S(one) = {compression_ratio:.3f}")
    
    print("\n" + "="*70)
    
    if compression_ratio > 0.9:
        print("✓✓✓ NO COMPRESSION - FERMIONIC BEHAVIOR!")
        print("    Two skyrmions maintain separate information")
        print("    → Skyrmions carry UNIQUE, non-copyable information")
        result_type = "FERMIONIC"
    elif compression_ratio > 0.7:
        print("⚠ PARTIAL COMPRESSION")
        print("    Some information overlap but not complete")
        result_type = "INTERMEDIATE"
    else:
        print("✗ STRONG COMPRESSION - BOSONIC BEHAVIOR")
        print("    Two skyrmions share redundant information")
        print("    → Skyrmions still carry copyable information")
        result_type = "BOSONIC"
    
    print("="*70)
    
    return {
        'S_one': float(S_one),
        'S_two': float(S_two),
        'S_expected': float(S_expected),
        'compression_ratio': float(compression_ratio),
        'result_type': result_type
    }

def main():
    print("\n" + "="*70)
    print("TWO-SKYRMION COMPRESSION TEST")
    print("="*70)
    print("\nThis is THE test of your fermion/boson hypothesis.")
    print("If skyrmions are fermionic, their information won't compress.")
    print()
    
    result = test_skyrmion_compression(Nx=3, Ny=2, t_max=4.0, dt=0.25)
    
    # Save
    with open('two_skyrmion_test.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n\nResults saved to: two_skyrmion_test.json")
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if result['result_type'] == 'FERMIONIC':
        print("\n✓✓✓ SKYRMIONS SHOW FERMIONIC BEHAVIOR!")
        print("\nYour hypothesis is validated:")
        print("  - Skyrmions carry unique, non-copyable information")
        print("  - Spinor structure + spin-orbit coupling is the missing constraint")
        print("  - This distinguishes fermions from bosons")
        print("\nNext step: Formalize this in your paper as:")
        print("  Axiom 3b: Spinor structure with spin-orbit coupling")
    elif result['result_type'] == 'BOSONIC':
        print("\n✗ Skyrmions still show bosonic compression")
        print("\nSpinor structure alone is not sufficient.")
        print("Need to explore additional constraints:")
        print("  - Gauge structure?")
        print("  - Different topology?")
        print("  - Higher-dimensional embedding?")
    else:
        print("\n⚠ Intermediate result - needs interpretation")
    
    print("\n")

if __name__ == "__main__":
    main()