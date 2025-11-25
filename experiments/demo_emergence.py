#!/usr/bin/env python3
"""
Quick demonstration of emergent fermionic statistics.

Run: python demo_emergence.py
"""

import numpy as np
from spin_statistics_engine import (
    SpinfulParticleSpace,
    build_spinful_hamiltonian,
    find_ground_states,
    full_analysis,
    analyze_spin_state
)

np.set_printoptions(precision=4, suppress=True)


def main():
    print("="*70)
    print("EMERGENT FERMIONIC STATISTICS: DEMONSTRATION")
    print("="*70)
    print()
    
    n_sites = 4
    t = 1.0
    U = 5.0
    
    space = SpinfulParticleSpace(n_sites, n_particles=2)
    print(f"System: {n_sites} sites, 2 particles with spin-1/2")
    print(f"Hilbert space dimension: {space.dim}")
    print()
    
    # Scan over spin coupling
    print("Scanning spin coupling J_spin:")
    print()
    print("J_spin   Exchange   Antisym   Singlet   Character")
    print("-" * 55)
    
    for J in [-2, -1, 0, 0.5, 1, 2, 5]:
        H = build_spinful_hamiltonian(space, t=t, U=U, J_spin=J)
        E, psi = find_ground_states(H, k=1)
        gs = psi[:, 0]
        
        analysis = full_analysis(gs, space)
        spin = analyze_spin_state(gs, space)
        
        exch = np.real(analysis['exchange_eigenvalue'])
        anti = analysis['antisymmetric_weight']
        sing = spin['singlet_weight']
        char = analysis['character']
        
        print(f"{J:6.1f}   {exch:8.4f}   {anti:7.4f}   {sing:7.4f}   {char}")
    
    print()
    print("="*70)
    print("KEY INSIGHT")
    print("="*70)
    print()
    print("With J > 0 (antiferromagnetic):")
    print("  - Ground state becomes ANTISYMMETRIC (exchange = -1)")
    print("  - Spin state is SINGLET (antisymmetric in spin)")
    print("  - This is TRUE FERMIONIC behavior!")
    print()
    print("The antisymmetry EMERGES from spin dynamics.")
    print("We didn't assume it - it arose from the Hamiltonian structure.")
    print()
    
    # Verify Pauli exclusion in detail
    print("="*70)
    print("PAULI EXCLUSION CHECK")
    print("="*70)
    print()
    
    H = build_spinful_hamiltonian(space, t=t, U=U, J_spin=2.0)
    E, psi = find_ground_states(H, k=1)
    gs = psi[:, 0]
    
    same_site_same_spin = 0.0
    same_site_diff_spin = 0.0
    diff_site = 0.0
    
    for idx, state in enumerate(space.states):
        prob = abs(gs[idx])**2
        x1, s1, x2, s2 = state
        
        if x1 == x2:
            if s1 == s2:
                same_site_same_spin += prob
            else:
                same_site_diff_spin += prob
        else:
            diff_site += prob
    
    print(f"Probability of same site, SAME spin:   {same_site_same_spin:.6f}")
    print(f"Probability of same site, DIFF spin:   {same_site_diff_spin:.6f}")
    print(f"Probability of different sites:        {diff_site:.6f}")
    print()
    
    if same_site_same_spin < 1e-10:
        print("✓ PAULI EXCLUSION VERIFIED!")
        print("  Two particles with same spin CANNOT be at same site.")
    else:
        print("✗ Some violation detected.")
    
    print()
    print("="*70)
    print("RUN STABILITY TESTS")
    print("="*70)
    print()
    print("To verify sectors are dynamically stable, run:")
    print("  python test_dynamical_stability.py")
    print()
    print("To run parameter sweep:")
    print("  python test_dynamical_stability.py --sweep")
    print()


if __name__ == '__main__':
    main()