#!/usr/bin/env python
"""
test_spinor_compression.py

Test whether SPINOR patterns with different spin textures develop different
entanglement structures.

Key test: Do skyrmions (spin texture) entangle differently than plain vortices?

Measurement: Bipartite entanglement between left and right halves of lattice.
- Higher S = more entanglement spread across system
- Different patterns should show different entanglement if spinor structure matters
"""

import numpy as np
import json
from qutip import entropy_vn
from quantum_spinor_lattice import SpinorLattice

def test_single_spinor(winding_spatial, winding_spin, Nx=2, Ny=2, 
                       t_max=2.0, dt=0.2):
    """Evolve single spinor pattern and measure entropy."""
    print(f"\n[TEST] Single spinor: w_spatial={winding_spatial}, w_spin={winding_spin}")
    
    lattice = SpinorLattice(Nx, Ny)
    lattice.build_hamiltonian(J_hop=1.0, J_spin=0.5, g_spinorbit=0.3, mass=0.1)
    lattice.init_spinor_vortex(Nx/2, Ny/2, winding_spatial, winding_spin)
    
    result = lattice.evolve(t_max, dt)
    
    # Measure entropy of HALF the system (trace out the other half)
    # This measures entanglement between left and right halves
    left_sites = [0, 2]  # Left column
    right_sites = [1, 3]  # Right column
    
    final_state = result.states[-1]
    
    # Entropy of left half = entanglement with right half
    rho_left = final_state.ptrace(left_sites)
    S_left = entropy_vn(rho_left)
    
    # Entropy of right half = entanglement with left half (should be same)
    rho_right = final_state.ptrace(right_sites)
    S_right = entropy_vn(rho_right)
    
    print(f"  Entanglement entropy: S_left = {S_left:.6f}, S_right = {S_right:.6f}")
    
    return S_left

def compare_spinor_types():
    """
    Compare different types of spinor patterns:
    1. No spin texture (w_spin=0) - like scalar
    2. Spin texture matches spatial (w_spin=w_spatial) - skyrmion
    3. Opposite spin texture (w_spin=-w_spatial) - anti-skyrmion
    """
    print("="*70)
    print("COMPARING SPINOR PATTERNS")
    print("="*70)
    
    results = {}
    
    # Test configurations
    configs = [
        (0, 0, "w=0, no spin texture (bosonic?)"),
        (1, 0, "w=1, no spin texture (spatial only)"),
        (1, 1, "w=1, matching spin texture (skyrmion)"),
        (1, -1, "w=1, opposite spin texture (anti-skyrmion)"),
    ]
    
    for w_spatial, w_spin, description in configs:
        print(f"\n--- {description} ---")
        try:
            S = test_single_spinor(w_spatial, w_spin, Nx=2, Ny=2, t_max=5.0, dt=0.25)
            results[f"w{w_spatial}_s{w_spin}"] = {
                'w_spatial': w_spatial,
                'w_spin': w_spin,
                'entropy': float(S),
                'description': description
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if len(results) > 0:
        for key, data in results.items():
            print(f"{data['description']:50s} S = {data['entropy']:.4f}")
        
        # Check if spin texture matters
        if 'w1_s0' in results and 'w1_s1' in results:
            S_no_spin = results['w1_s0']['entropy']
            S_skyrmion = results['w1_s1']['entropy']
            diff = abs(S_skyrmion - S_no_spin)
            
            print(f"\nSpin texture effect: ΔS = {diff:.4f}")
            
            if diff > 0.1:
                print("✓ Spin texture significantly changes entanglement")
                print("  → Skyrmions build different entanglement structure!")
            else:
                print("⚠ Spin texture has minimal effect on entanglement")
                print("  → Spinor structure may not be sufficient constraint")
    
    return results

def main():
    print("\n" + "="*70)
    print("SPINOR PATTERN ENTANGLEMENT TEST")
    print("="*70)
    print("\nThis tests whether spinor structure (not just winding)")
    print("produces patterns with different ENTANGLEMENT properties.")
    print("\nWe measure bipartite entanglement (left-right split).")
    print("Higher S = more entanglement = pattern spread across system.")
    print()
    
    results = compare_spinor_types()
    
    # Save
    with open('spinor_compression_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: spinor_compression_results.json")
    print()

if __name__ == "__main__":
    main()