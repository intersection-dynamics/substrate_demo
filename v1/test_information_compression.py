#!/usr/bin/env python
"""
test_information_compression.py

Test whether two patterns with same winding compress their information (bosonic)
or maintain separate information (fermionic).

The hypothesis:
- Bosons: S(two patterns) < 2×S(one pattern) - information compresses
- Fermions: S(two patterns) ≈ 2×S(one pattern) - information doesn't compress

This tests whether patterns carry copyable (bosonic) or unique (fermionic) information.
"""

import numpy as np
import argparse
import os
import json

try:
    from qutip import Qobj, entropy_vn
    QUTIP_AVAILABLE = True
except ImportError:
    print("[ERROR] QuTiP required")
    import sys
    sys.exit(1)

from quantum_substrate_evolution import QuantumSubstrateEvolver

def measure_single_pattern_entropy(winding, Nx=3, Ny=3, t_max=3.0, dt=0.2,
                                   J_nn=1.0, J_defrag=0.5, mass=0.1):
    """
    Evolve a single pattern and measure its final entropy.
    """
    print(f"\n{'='*70}")
    print(f"SINGLE PATTERN TEST (w={winding})")
    print(f"{'='*70}")
    
    evolver = QuantumSubstrateEvolver(Nx, Ny)
    evolver.build_hamiltonian(J_nn=J_nn, J_defrag=J_defrag, mass=mass)
    
    # Single pattern at center
    evolver.init_state('single_vortex', x0=Nx/2, y0=Ny/2, winding=winding)
    
    # Pattern sites (center region)
    center_x, center_y = Nx/2, Ny/2
    pattern_sites = [evolver.site_index(i, j) 
                    for i in range(Nx) for j in range(Ny)
                    if (i - center_x)**2 + (j - center_y)**2 < 2.0]
    
    print(f"Pattern region: {len(pattern_sites)} sites: {pattern_sites}")
    
    # Evolve
    times = np.arange(0, t_max + dt, dt)
    result = evolver.evolve(t_max, dt, measurement_times=times)
    
    # Final state entropy
    final_state = result.states[-1]
    rho_pattern = final_state.ptrace(pattern_sites)
    S_single = entropy_vn(rho_pattern)
    
    print(f"Final pattern entropy: S(single) = {S_single:.6f}")
    
    return S_single, pattern_sites

def measure_two_pattern_entropy(winding, separation, Nx=3, Ny=3, t_max=3.0, dt=0.2,
                                J_nn=1.0, J_defrag=0.5, mass=0.1):
    """
    Evolve two patterns close together and measure their combined entropy.
    """
    print(f"\n{'='*70}")
    print(f"TWO PATTERN TEST (w={winding}, separation={separation})")
    print(f"{'='*70}")
    
    evolver = QuantumSubstrateEvolver(Nx, Ny)
    evolver.build_hamiltonian(J_nn=J_nn, J_defrag=J_defrag, mass=mass)
    
    # Two patterns separated by 'separation' lattice units
    x1 = Nx/2 - separation/2
    y1 = Ny/2
    x2 = Nx/2 + separation/2
    y2 = Ny/2
    
    print(f"Pattern 1 at ({x1:.1f}, {y1:.1f})")
    print(f"Pattern 2 at ({x2:.1f}, {y2:.1f})")
    
    evolver.init_state('two_vortex', x1=x1, y1=y1, x2=x2, y2=y2, winding=winding)
    
    # Combined pattern region (both patterns)
    pattern_sites = []
    for i in range(Nx):
        for j in range(Ny):
            dx1, dy1 = i - x1, j - y1
            dx2, dy2 = i - x2, j - y2
            if dx1**2 + dy1**2 < 2.0 or dx2**2 + dy2**2 < 2.0:
                pattern_sites.append(evolver.site_index(i, j))
    
    pattern_sites = sorted(list(set(pattern_sites)))
    print(f"Combined pattern region: {len(pattern_sites)} sites: {pattern_sites}")
    
    # Evolve
    times = np.arange(0, t_max + dt, dt)
    result = evolver.evolve(t_max, dt, measurement_times=times)
    
    # Final state entropy
    final_state = result.states[-1]
    rho_patterns = final_state.ptrace(pattern_sites)
    S_two = entropy_vn(rho_patterns)
    
    print(f"Final combined entropy: S(two) = {S_two:.6f}")
    
    return S_two, pattern_sites

def test_compression(winding, separation=1.0, Nx=3, Ny=3, t_max=3.0, dt=0.2,
                    J_nn=1.0, J_defrag=0.5, mass=0.1):
    """
    Test whether information compresses when two patterns overlap.
    """
    print("\n" + "="*70)
    print(f"INFORMATION COMPRESSION TEST")
    print(f"Winding number: {winding}")
    print(f"Lattice: {Nx}×{Ny}")
    print(f"Pattern separation: {separation} lattice units")
    print("="*70)
    
    # Measure single pattern
    S_single, single_sites = measure_single_pattern_entropy(
        winding, Nx, Ny, t_max, dt, J_nn, J_defrag, mass
    )
    
    # Measure two patterns
    S_two, two_sites = measure_two_pattern_entropy(
        winding, separation, Nx, Ny, t_max, dt, J_nn, J_defrag, mass
    )
    
    # Analysis
    print("\n" + "="*70)
    print("INFORMATION COMPRESSION ANALYSIS")
    print("="*70)
    
    S_expected_independent = 2 * S_single
    compression_ratio = S_two / S_expected_independent
    
    print(f"\nEntropy measurements:")
    print(f"  Single pattern:    S(one)  = {S_single:.6f}")
    print(f"  Two patterns:      S(two)  = {S_two:.6f}")
    print(f"  Expected if independent: 2×S(one) = {S_expected_independent:.6f}")
    
    print(f"\nCompression ratio: S(two) / (2×S(one)) = {compression_ratio:.3f}")
    
    # Interpretation
    print(f"\n{'='*70}")
    
    if compression_ratio < 0.7:
        print("✓ STRONG COMPRESSION (Bosonic behavior)")
        print("  Information from two patterns is highly redundant")
        print("  Patterns carry copyable/identical information")
        result_type = "bosonic"
    elif compression_ratio < 0.9:
        print("⚠ MODERATE COMPRESSION (Partial bosonic character)")
        print("  Some information overlap between patterns")
        result_type = "partial_bosonic"
    elif compression_ratio < 1.1:
        print("✓ NO COMPRESSION (Fermionic behavior)")
        print("  Two patterns maintain separate information")
        print("  Patterns carry unique/non-copyable information")
        result_type = "fermionic"
    else:
        print("? SUPERADDITIVE (Unexpected)")
        print("  S(two) > 2×S(one) - patterns became more entangled")
        result_type = "superadditive"
    
    print(f"{'='*70}\n")
    
    return {
        'winding': winding,
        'separation': separation,
        'S_single': float(S_single),
        'S_two': float(S_two),
        'S_expected': float(S_expected_independent),
        'compression_ratio': float(compression_ratio),
        'result_type': result_type,
        'single_pattern_sites': single_sites,
        'two_pattern_sites': two_sites
    }

def compare_windings(windings=[0, 1], separation=1.0, Nx=3, Ny=3, t_max=3.0, dt=0.2):
    """
    Compare compression behavior across different winding numbers.
    """
    print("\n" + "="*70)
    print("COMPARING WINDING NUMBERS")
    print("="*70)
    
    results = {}
    
    for w in windings:
        try:
            result = test_compression(w, separation, Nx, Ny, t_max, dt)
            results[w] = result
        except Exception as e:
            print(f"\n✗ Error testing w={w}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        print()
        
        for w, result in results.items():
            print(f"w = {w:2d}: Compression ratio = {result['compression_ratio']:.3f} → {result['result_type']}")
        
        print()
        
        # Check if w=±1 behave differently from w=0
        if 0 in results and 1 in results:
            w0_ratio = results[0]['compression_ratio']
            w1_ratio = results[1]['compression_ratio']
            
            if w1_ratio > w0_ratio + 0.2:
                print("✓ w=±1 shows LESS compression than w=0")
                print("  → Consistent with w=±1 being fermionic, w=0 bosonic")
            elif w1_ratio < w0_ratio - 0.2:
                print("✗ w=±1 shows MORE compression than w=0")
                print("  → Opposite of expected fermion/boson distinction")
            else:
                print("⚠ w=±1 and w=0 show similar compression")
                print("  → No clear fermion/boson distinction in compression")
        
        print()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test information compression")
    parser.add_argument("--winding", type=int, default=None, help="Single winding to test")
    parser.add_argument("--compare", action="store_true", help="Compare w=0,1,-1")
    parser.add_argument("--separation", type=float, default=1.0, help="Pattern separation")
    parser.add_argument("--Nx", type=int, default=3)
    parser.add_argument("--Ny", type=int, default=3)
    parser.add_argument("--t_max", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="compression_results.json")
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_windings([0, 1, -1], args.separation, args.Nx, args.Ny, 
                                  args.t_max, args.dt)
    elif args.winding is not None:
        result = test_compression(args.winding, args.separation, args.Nx, args.Ny,
                                 args.t_max, args.dt)
        results = {args.winding: result}
    else:
        print("Specify --winding N or --compare")
        return
    
    # Save results
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.out}")

if __name__ == "__main__":
    main()